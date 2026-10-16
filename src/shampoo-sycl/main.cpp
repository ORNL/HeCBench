#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>
#include <sycl/sycl.hpp>
// Full fp32 products, matching cuBLAS/hipBLAS pedantic math. standard
// disallows TF32 and bf16 GEMMs and overrides MKL_BLAS_COMPUTE_MODE from
// the environment. This Newton-Schulz iteration diverges under those modes.
#ifndef MKL_BLAS_COMPUTE_MODE
#define MKL_BLAS_COMPUTE_MODE oneapi::mkl::blas::compute_mode::standard
#endif
#include <oneapi/mkl.hpp>
#include "reference.h"

// One order-2 root-inverse Shampoo step on a single blocked parameter of
// shape (rows x cols). Factor matrices stay in fp32. The inverse fourth root
// is the coupled Newton-Schulz iteration (Polar Express coefficients).
//
// https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/preconditioner/shampoo_preconditioner_list.py
//
// The search direction is then rescaled to the Frobenius norm of an AdaGrad
// direction on the same block (Shi et al., arXiv:2309.06497, Section 2.4).
// That is layer-wise learning-rate grafting: Shampoo chooses the direction
// of this block's weight update, and AdaGrad supplies its length.

using transpose = oneapi::mkl::transpose;

// 256-thread kernels. Contiguous stages move 128 bits per thread.
static const int kThreads = 256;

// Number of 256-thread groups that cover `count` floats four at a time.
// A count below 4 still yields one group. Item 0 of that group handles
// the leftover entries inside the kernels.
static int blocks_for4(int count) {
  int n4 = count >> 2;
  int blocks = (n4 + kThreads - 1) / kThreads;
  return blocks > 0 ? blocks : 1;
}

// Row-major C = alpha * op(A) @ op(B) + beta * C. C is M x N and K is the
// inner dimension. oneMKL is column-major, so the launch is the same product
// on the transposed views: C^T from op(B)^T and op(A)^T.
static void rgemm(sycl::queue& q, transpose opA, transpose opB,
                  int M, int N, int K,
                  const float* A, const float* B, float* C,
                  float alpha, float beta) {
  const int lda = (opA == transpose::nontrans) ? K : M;
  const int ldb = (opB == transpose::nontrans) ? N : K;
  const int ldc = N;
  oneapi::mkl::blas::column_major::gemm(
      q, opB, opA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc,
      oneapi::mkl::blas::compute_mode::standard);
}

// Launch range for a kernel that consumes `count` floats four at a time:
// one work-item per four elements, rounded up to a multiple of 256.
static sycl::nd_range<1> nd4(int count) {
  return sycl::nd_range<1>(blocks_for4(count) * kThreads, kThreads);
}

// Write ||A||_F to *nrm. The sum of squares runs in double, then one square
// root converts it to the norm. One work-group. Item t owns float4 slots
// t, t+256, ....
// The inverse root divides by this norm, and the ridge is proportional to it.
static void frobenius_nrm(sycl::queue& q, const float* A, float* nrm, int count) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<double, 1> smem(kThreads, h);
    h.parallel_for(sycl::nd_range<1>(kThreads, kThreads),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(256)]] {
      const int tid = it.get_local_id(0);
      const int n4 = count >> 2;
      const sycl::float4* A4 = reinterpret_cast<const sycl::float4*>(A);
      double sum = 0.0;
      for (int i = tid; i < n4; i += kThreads) {
        sycl::float4 v = A4[i];
        sum += (double)v.x() * (double)v.x();
        sum += (double)v.y() * (double)v.y();
        sum += (double)v.z() * (double)v.z();
        sum += (double)v.w() * (double)v.w();
      }
      if (tid == 0) {
        for (int j = n4 * 4; j < count; ++j)
          sum += (double)A[j] * (double)A[j];
      }
      smem[tid] = sum;
      it.barrier(sycl::access::fence_space::local_space);
      for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        it.barrier(sycl::access::fence_space::local_space);
      }
      if (tid == 0) *nrm = (float)sycl::sqrt(smem[0]);
    });
  });
}

// Adds the epsilon-I regularization in the factors of paper Eqs. (14)-(15).
// This implementation uses max(||A||_F * rel_eps, eps), following the cited
// upstream Newton-Schulz solver. A rectangular gradient has a singular Gram
// matrix, so this ridge keeps its inverse root finite.
static void add_ridge(sycl::queue& q, float* A, const float* nrm,
                      float rel_eps, float eps, int n) {
  const int blocks = (n + kThreads - 1) / kThreads;
  q.parallel_for(sycl::nd_range<1>(blocks * kThreads, kThreads),
                 [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(256)]] {
    int i = it.get_global_id(0);
    if (i < n) A[(size_t)i * n + i] += sycl::fmax((*nrm) * rel_eps, eps);
  });
}

// Z_ii = 1. The caller zeros Z first, so Z becomes I, the starting value of
// the inverse-square-root side of the coupled Newton-Schulz iteration.
static void set_diag(sycl::queue& q, float* Z, int n) {
  const int blocks = (n + kThreads - 1) / kThreads;
  q.parallel_for(sycl::nd_range<1>(blocks * kThreads, kThreads),
                 [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(256)]] {
    int i = it.get_global_id(0);
    if (i < n) Z[(size_t)i * n + i] = 1.f;
  });
}

// Bias-correct the EMA factors following paper Section 3.1.1 after
// Eqs. (25)-(26): Lhat=L/(1-beta2^step), and likewise for R.
// The denominator is a host value because each step relaunches the kernels.
static void scale_div(sycl::queue& q, float* dst, const float* src,
                      float denom, int count) {
  q.parallel_for(nd4(count), [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(256)]] {
    const int n4 = count >> 2;
    const int i = it.get_global_id(0);
    if (i < n4) {
      sycl::float4 v = reinterpret_cast<const sycl::float4*>(src)[i];
      v.x() /= denom; v.y() /= denom; v.z() /= denom; v.w() /= denom;
      reinterpret_cast<sycl::float4*>(dst)[i] = v;
    }
    if (i == 0)
      for (int j = n4 * 4; j < count; ++j) dst[j] = src[j] / denom;
  });
}

// Y = a * Y + b * X, entrywise. For factor updates this implements paper
// Eqs. (25)-(26), with a=beta2, b=1-beta2, and X equal to GG^T or G^T G.
// Newton-Schulz also reuses it for its coupled polynomial; that solver is
// from the cited upstream implementation, not an iteration in this paper.
static void axpy(sycl::queue& q, float* Y, const float* X,
                 float a, float b, int count) {
  q.parallel_for(nd4(count), [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(256)]] {
    const int n4 = count >> 2;
    const int i = it.get_global_id(0);
    if (i < n4) {
      sycl::float4 y = reinterpret_cast<sycl::float4*>(Y)[i];
      sycl::float4 x = reinterpret_cast<const sycl::float4*>(X)[i];
      y.x() = a * y.x() + b * x.x();
      y.y() = a * y.y() + b * x.y();
      y.z() = a * y.z() + b * x.z();
      y.w() = a * y.w() + b * x.w();
      reinterpret_cast<sycl::float4*>(Y)[i] = y;
    }
    if (i == 0)
      for (int j = n4 * 4; j < count; ++j) Y[j] = a * Y[j] + b * X[j];
  });
}

// Paper Section 2.2, Eq. (9), diagonal AdaGrad, stored with G's shape:
//   A_t = A_{t-1} + G_t^{odot 2}.
// Section 2.4 and Algorithm 1 use this diagonal state for grafting.
static void square_accum(sycl::queue& q, float* A, const float* G, int count) {
  q.parallel_for(nd4(count), [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(256)]] {
    const int n4 = count >> 2;
    const int i = it.get_global_id(0);
    if (i < n4) {
      sycl::float4 a = reinterpret_cast<sycl::float4*>(A)[i];
      sycl::float4 g = reinterpret_cast<const sycl::float4*>(G)[i];
      a.x() += g.x() * g.x();
      a.y() += g.y() * g.y();
      a.z() += g.z() * g.z();
      a.w() += g.w() * g.w();
      reinterpret_cast<sycl::float4*>(A)[i] = a;
    }
    if (i == 0)
      for (int j = n4 * 4; j < count; ++j) A[j] += G[j] * G[j];
  });
}

// Paper Section 2.2, Eq. (13), the diagonal AdaGrad direction. Eq. (11) is
// the same componentwise division before epsilon regularization:
//   P_graft = G_t / (sqrt(A_t) + epsilon).
// Section 2.4 and Algorithm 1 use this direction as the graft.
static void graft_direction(sycl::queue& q, float* dst, const float* G,
                            const float* A, float eps, int count) {
  q.parallel_for(nd4(count), [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(256)]] {
    const int n4 = count >> 2;
    const int i = it.get_global_id(0);
    if (i < n4) {
      sycl::float4 g = reinterpret_cast<const sycl::float4*>(G)[i];
      sycl::float4 a = reinterpret_cast<const sycl::float4*>(A)[i];
      g.x() /= sycl::sqrt(a.x()) + eps;
      g.y() /= sycl::sqrt(a.y()) + eps;
      g.z() /= sycl::sqrt(a.z()) + eps;
      g.w() /= sycl::sqrt(a.w()) + eps;
      reinterpret_cast<sycl::float4*>(dst)[i] = g;
    }
    if (i == 0)
      for (int j = n4 * 4; j < count; ++j)
        dst[j] = G[j] / (sycl::sqrt(A[j]) + eps);
  });
}

// Paper Section 2.4, Eq. (20): layer-wise learning-rate grafting.
//   P_shampoo *= ||P_graft||_F / ||P_shampoo||_F.
// Shampoo has chosen which way this block's weights move. The Frobenius
// norm is the distance they would travel if the matrix were unrolled into a
// vector. This kernel keeps that direction and replaces the distance with
// the diagonal AdaGrad step length on the same block. Every weight in the
// block is scaled by the same factor; another layer gets its own factor,
// which is the block-diagonal D_t in Eqs. (22)-(23). A learning rate tuned
// for AdaGrad therefore still means the same distance. The minus sign and
// learning rate belong to the parameter update, which this benchmark does
// not perform. 1e-16 guards a zero Shampoo direction.
static void scale_by_graft(sycl::queue& q, float* Y, const float* nrm_graft,
                           const float* nrm_shampoo, int count) {
  q.parallel_for(nd4(count), [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(256)]] {
    const float scale = (*nrm_graft) / ((*nrm_shampoo) + kGraftNormEps);
    const int n4 = count >> 2;
    const int i = it.get_global_id(0);
    if (i < n4) {
      sycl::float4 y = reinterpret_cast<sycl::float4*>(Y)[i];
      y.x() *= scale; y.y() *= scale; y.z() *= scale; y.w() *= scale;
      reinterpret_cast<sycl::float4*>(Y)[i] = y;
    }
    if (i == 0)
      for (int j = n4 * 4; j < count; ++j) Y[j] *= scale;
  });
}

// Y = A / max(*nrm, 1e-8), entrywise. *nrm is ||A||_F.
// Dividing by ||A||_F puts every singular value in [0, 1], the range the
// Polar Express coefficients were fit for. The floor avoids a zero divisor.
static void div_by_norm(sycl::queue& q, float* Y, const float* A,
                        const float* nrm, int count) {
  q.parallel_for(nd4(count), [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(256)]] {
    const float alpha = sycl::fmax(*nrm, 1e-8f);
    const int n4 = count >> 2;
    const int i = it.get_global_id(0);
    if (i < n4) {
      sycl::float4 v = reinterpret_cast<const sycl::float4*>(A)[i];
      v.x() /= alpha; v.y() /= alpha; v.z() /= alpha; v.w() /= alpha;
      reinterpret_cast<sycl::float4*>(Y)[i] = v;
    }
    if (i == 0)
      for (int j = n4 * 4; j < count; ++j) Y[j] = A[j] / alpha;
  });
}

// Undo the Frobenius normalization after one Newton-Schulz pass.
// s = sqrt(max(*nrm, 1e-8)), and *nrm is ||A||_F from before the division.
// mode 0 writes src * s. Applied to Y, that restores a square root.
// mode 1 writes src / s. Applied to Z, that restores an inverse square root.
static void scale_from_norm(sycl::queue& q, float* dst, const float* src,
                            const float* nrm, int count, int mode) {
  q.parallel_for(nd4(count), [=](sycl::nd_item<1> it) [[sycl::reqd_work_group_size(256)]] {
    const float alpha = sycl::fmax(*nrm, 1e-8f);
    const float s = sycl::sqrt(alpha);
    const int n4 = count >> 2;
    const int i = it.get_global_id(0);
    if (i < n4) {
      sycl::float4 v = reinterpret_cast<const sycl::float4*>(src)[i];
      if (mode == 0) { v.x() *= s; v.y() *= s; v.z() *= s; v.w() *= s; }
      else           { v.x() /= s; v.y() /= s; v.z() /= s; v.w() /= s; }
      reinterpret_cast<sycl::float4*>(dst)[i] = v;
    }
    if (i == 0) {
      for (int j = n4 * 4; j < count; ++j)
        dst[j] = (mode == 0) ? src[j] * s : src[j] / s;
    }
  });
}

// Overwrite the n x n matrix A with A^{-1/4}. Y, Z, T, and B are n x n
// workspaces. nrm is a one-float device scalar.
//
// Add the relative ridge, then run one coupled pass per factor of two in
// kShampooRoot. Each pass sets Y = A / ||A||_F and Z = I, then applies the
// ten Polar Express quintics
//   T = Z Y,  B = b T + c T^2,  Y = a Y + Y B,  Z = a Z + B Z.
// Y moves toward the square root of the normalized matrix and Z toward its
// inverse square root. Every pass but the last writes A = sqrt(||A||_F) * Y,
// leaving a square root. The last pass writes A = Z / sqrt(||A||_F), leaving
// an inverse square root of whatever entered that pass. Root 4 is two
// passes, so the second turns A^{1/2} into A^{-1/4}.
static void inverse_root(sycl::queue& q, float* A, float* Y, float* Z,
                         float* T, float* B, float* nrm, int n) {
  const int count = n * n;
  frobenius_nrm(q, A, nrm, count);
  add_ridge(q, A, nrm, kRelativeEpsilon, kEpsilon, n);

  int passes = 0;
  for (int r = kShampooRoot; r > 1; r >>= 1) passes++;
  for (int pass = passes; pass >= 1; pass--) {
    frobenius_nrm(q, A, nrm, count);
    div_by_norm(q, Y, A, nrm, count);
    q.memset(Z, 0, sizeof(float) * (size_t)count);
    set_diag(q, Z, n);
    for (int s = 0; s < kNsIters; s++) {
      float a = kNs[s][0], b = kNs[s][1], c = kNs[s][2];
      rgemm(q, transpose::nontrans, transpose::nontrans, n, n, n, Z, Y, T, 1.f, 0.f);
      rgemm(q, transpose::nontrans, transpose::nontrans, n, n, n, T, T, B, c, 0.f);
      axpy(q, B, T, 1.f, b, count);
      rgemm(q, transpose::nontrans, transpose::nontrans, n, n, n, Y, B, T, 1.f, 0.f);
      axpy(q, Y, T, a, 1.f, count);
      rgemm(q, transpose::nontrans, transpose::nontrans, n, n, n, B, Z, T, 1.f, 0.f);
      axpy(q, Z, T, a, 1.f, count);
    }
    const float* src = (pass > 1) ? Y : Z;
    scale_from_norm(q, A, src, nrm, count, pass > 1 ? 0 : 1);
  }
}

// Time `steps` Shampoo updates of one frozen M x N gradient. verify != 0
// compares the preconditioned gradient with the host reference.
int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <rows> <cols> <steps> <verify>\n", argv[0]);
    printf("  rows x cols is one order-2 Shampoo block\n");
    return 1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int steps = atoi(argv[3]);
  const int verify = atoi(argv[4]);
  if (M < 1 || N < 1 || M > 8192 || N > 8192 || steps < 1) {
    fprintf(stderr, "rows and cols must be in [1, 8192], steps >= 1\n");
    return 1;
  }

  const int lcount = M * M;
  const int rcount = N * N;
  const int gcount = M * N;
  const int nmax = std::max(M, N);

  float* G = (float*)malloc(sizeof(float) * (size_t)gcount);
  float* out = (float*)malloc(sizeof(float) * (size_t)gcount);
  float* H = nullptr;
  float* Gs = nullptr;
  float* pre = nullptr;
  float* Lh = nullptr;
  float* Rh = nullptr;
  float* ref_pre = nullptr;
  float* ref_L = nullptr;
  float* ref_R = nullptr;
  float* ref = nullptr;
  if (verify) {
    H = (float*)malloc(sizeof(float) * (size_t)gcount);
    Gs = (float*)malloc(sizeof(float) * (size_t)gcount);
    pre = (float*)malloc(sizeof(float) * (size_t)gcount);
    Lh = (float*)malloc(sizeof(float) * (size_t)lcount);
    Rh = (float*)malloc(sizeof(float) * (size_t)rcount);
    ref_pre = (float*)malloc(sizeof(float) * (size_t)gcount);
    ref_L = (float*)malloc(sizeof(float) * (size_t)lcount);
    ref_R = (float*)malloc(sizeof(float) * (size_t)rcount);
    ref = (float*)malloc(sizeof(float) * (size_t)gcount);
  }
  if (!G || !out || (verify && (!H || !Gs || !pre || !Lh || !Rh || !ref_pre ||
                                !ref_L || !ref_R || !ref))) {
    fprintf(stderr, "host allocation failed\n");
    return 1;
  }

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (int i = 0; i < gcount; i++) G[i] = dist(gen);
  // H is a second pattern, not a scale of G. Each step reads G + (step-1)*H,
  // so the Gram matrix changes shape and a wrong beta2 or bias correction
  // cannot hide behind grafting.
  if (verify)
    for (int i = 0; i < gcount; i++) H[i] = dist(gen);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float* d_G = sycl::malloc_device<float>(gcount, q);
  float* d_L = sycl::malloc_device<float>(lcount, q);
  float* d_R = sycl::malloc_device<float>(rcount, q);
  float* d_Linv = sycl::malloc_device<float>(lcount, q);
  float* d_Rinv = sycl::malloc_device<float>(rcount, q);
  float* d_out = sycl::malloc_device<float>(gcount, q);
  float* d_A = sycl::malloc_device<float>(gcount, q);
  float* d_Y = sycl::malloc_device<float>((size_t)nmax * nmax, q);
  float* d_Z = sycl::malloc_device<float>((size_t)nmax * nmax, q);
  float* d_T = sycl::malloc_device<float>((size_t)nmax * nmax, q);
  float* d_B = sycl::malloc_device<float>((size_t)nmax * nmax, q);
  // Two scalars: ||P_shampoo||_F and ||P_graft||_F. The inverse root uses
  // the first slot and finishes before grafting writes either one.
  float* d_nrm = sycl::malloc_device<float>(2, q);
  if (!d_G || !d_L || !d_R || !d_Linv || !d_Rinv || !d_out || !d_A ||
      !d_Y || !d_Z || !d_T || !d_B || !d_nrm) {
    fprintf(stderr, "device allocation failed\n");
    return 1;
  }

  q.memcpy(d_G, G, sizeof(float) * (size_t)gcount);
  q.memset(d_L, 0, sizeof(float) * (size_t)lcount);
  q.memset(d_R, 0, sizeof(float) * (size_t)rcount);
  q.memset(d_A, 0, sizeof(float) * (size_t)gcount);
  q.wait();

  auto start = std::chrono::steady_clock::now();

  // One step on d_G. Verification replaces d_G each step; timing leaves it
  // fixed. L and R are the row and column factors. Launched directly each
  // step, as on CUDA and HIP. bc is the host bias
  // correction 1 - beta2^step.
  //   Eqs. (25)-(26): EMA updates of L from GG^T and R from G^T G
  //   Section 3.1.1: bias correction, then Algorithm 1: inverse fourth roots
  //   Eq. (17): out = Linv^T G Rinv (Linv is symmetric)
  //   Section 2.2, Eqs. (9) and (13): diagonal AdaGrad state and direction
  //   Eq. (20): rescale out by ||P_graft||_F / ||out||_F
  for (int step = 1; step <= steps; step++) {
    if (verify) {
      shampoo_gradient_at_step(G, H, Gs, gcount, step);
      q.memcpy(d_G, Gs, sizeof(float) * (size_t)gcount);
    }
    rgemm(q, transpose::nontrans, transpose::trans, M, M, N, d_G, d_G, d_T, 1.f, 0.f);
    axpy(q, d_L, d_T, kBeta2, kWeight, lcount);
    rgemm(q, transpose::trans, transpose::nontrans, N, N, M, d_G, d_G, d_T, 1.f, 0.f);
    axpy(q, d_R, d_T, kBeta2, kWeight, rcount);

    float bc = shampoo_bias_correction2(step);
    scale_div(q, d_Linv, d_L, bc, lcount);
    inverse_root(q, d_Linv, d_Y, d_Z, d_T, d_B, d_nrm, M);
    scale_div(q, d_Rinv, d_R, bc, rcount);
    inverse_root(q, d_Rinv, d_Y, d_Z, d_T, d_B, d_nrm, N);

    rgemm(q, transpose::trans, transpose::nontrans, M, N, M, d_Linv, d_G, d_T, 1.f, 0.f);
    rgemm(q, transpose::nontrans, transpose::nontrans, M, N, N, d_T, d_Rinv, d_out, 1.f, 0.f);
    if (verify && step == steps)
      q.memcpy(pre, d_out, sizeof(float) * (size_t)gcount);

    square_accum(q, d_A, d_G, gcount);
    graft_direction(q, d_T, d_G, d_A, kGraftEpsilon, gcount);
    frobenius_nrm(q, d_out, d_nrm, gcount);
    frobenius_nrm(q, d_T, d_nrm + 1, gcount);
    scale_by_graft(q, d_out, d_nrm + 1, d_nrm, gcount);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  printf("rows %d cols %d steps %d\n", M, N, steps);
  printf("Average shampoo step time %f (ms)\n", ms / steps);

  q.memcpy(out, d_out, sizeof(float) * (size_t)gcount);
  if (verify) {
    q.memcpy(Lh, d_L, sizeof(float) * (size_t)lcount);
    q.memcpy(Rh, d_R, sizeof(float) * (size_t)rcount);
  }
  q.wait();
  sycl::free(d_G, q); sycl::free(d_L, q); sycl::free(d_R, q);
  sycl::free(d_Linv, q); sycl::free(d_Rinv, q); sycl::free(d_out, q);
  sycl::free(d_A, q);
  sycl::free(d_Y, q); sycl::free(d_Z, q); sycl::free(d_T, q);
  sycl::free(d_B, q); sycl::free(d_nrm, q);

  int rc = 0;
  if (verify) {
    // Factors and the pre-graft block catch a wrong EMA, bias correction, or
    // inverse-root scale. Grafting forces the output norm, so the grafted
    // block alone cannot see those errors.
    int ok = ref_shampoo(G, H, ref_L, ref_R, ref_pre, ref, M, N, steps) == 0 &&
             shampoo_match("L", Lh, ref_L, lcount) &&
             shampoo_match("R", Rh, ref_R, rcount) &&
             shampoo_match("pre-graft", pre, ref_pre, gcount) &&
             shampoo_match("grafted", out, ref, gcount);
    printf("%s\n", ok ? "PASS" : "FAIL");
    rc = ok ? 0 : 1;
  }

  free(G);
  free(out);
  free(H);
  free(Gs);
  free(pre);
  free(Lh);
  free(Rh);
  free(ref_pre);
  free(ref_L);
  free(ref_R);
  free(ref);
  return rc;
}
