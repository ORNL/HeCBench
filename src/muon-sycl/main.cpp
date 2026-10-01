#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "../muon-cuda/reference.h"

// The Muon optimizer is applied to a single 2D hidden weight matrix of shape
// (rows x cols) == (out_features x in_features) of a Linear layer. The momentum
// buffer, the gradient and the weights are kept in float32; the Newton-Schulz
// orthogonalization runs in bfloat16 (with float32 accumulation), as in
// the reference implementation https://github.com/KellerJordan/Muon .
//
// This version uses oneMKL bf16 GEMMs with fp32 accumulation for the
// Newton-Schulz matmuls, and fuses the elementwise steps.

using bf16 = sycl::ext::oneapi::bfloat16;

// row-major C(M x N) = opA(A) @ opB(B), bf16 storage, fp32 accumulation.
// oneMKL is column-major, so we compute C^T = opB(B)^T @ opA(A)^T by swapping
// the operands; a row-major (r x c) array is a column-major (c x r) array.
static inline void rgemm(sycl::queue &q,
                         oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                         int M, int N, int K,
                         const bf16* A, const bf16* B, bf16* C)
{
  const float alpha = 1.f, beta = 0.f;
  const int lda = (opA == oneapi::mkl::transpose::nontrans) ? K : M;
  const int ldb = (opB == oneapi::mkl::transpose::nontrans) ? N : K;
  const int ldc = N;
  oneapi::mkl::blas::column_major::gemm(q, opB, opA, N, M, K,
               alpha,
               B, ldb,
               A, lda,
               beta,
               C, ldc);
}

//   mom = beta*mom + (1-beta)*g; out = bf16((1-beta)*g + beta*mom)
static inline bf16 mom_step(float& m, float g, float beta)
{
  float t = (1.f - beta) * g;
  float mi = beta * m + t;
  m = mi;
  return bf16(t + beta * mi);
}

//        p = p*(1 - lr*wd) - lr*update
static inline float p_step(float p, float upd,
                           float decay, float lr, float scale)
{
  return p * decay - lr * (scale * upd);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <rows> <cols> <repeat> <verify>\n", argv[0]);
    printf("  rows = out_features, cols = in_features of a 2D weight matrix\n");
    return 1;
  }

  const int P = atoi(argv[1]);
  const int Q = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  const int verify = atoi(argv[4]);

  const int N = P * Q;
  const int mm_ = std::min(P, Q); // rows of the Newton-Schulz working matrix
  const int nn_ = std::max(P, Q); // cols of the Newton-Schulz working matrix

  const size_t size_bytes = (size_t)N * sizeof(float);

  float *mom = (float*) malloc (size_bytes);
  float *g   = (float*) malloc (size_bytes);
  float *p   = (float*) malloc (size_bytes);
  float *r   = (float*) malloc (size_bytes);   // reference param
  float *rm  = (float*) malloc (size_bytes);   // reference momentum

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (int i = 0; i < N; i++) {
    rm[i] = mom[i] = dist(gen);
    g[i] = dist(gen);
    r[i] = p[i] = dist(gen);
  }

  // Muon hyper-parameters
  const float beta = 0.95f;
  const float lr   = 0.02f;
  const float wd   = 0.f;     // weight decay (Muon default)
  const float scale = sqrtf(fmaxf(1.f, (float)P / (float)Q));

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_p  = sycl::malloc_device<float>(N, q);
  float *d_m  = sycl::malloc_device<float>(N, q);
  float *d_g  = sycl::malloc_device<float>(N, q);
  float *d_ss = sycl::malloc_device<float>(1, q);
  bf16  *d_X  = sycl::malloc_device<bf16>(N, q);          // mm_ x nn_ = N
  bf16  *d_A  = sycl::malloc_device<bf16>(mm_ * mm_, q);
  bf16  *d_A2 = sycl::malloc_device<bf16>(mm_ * mm_, q);
  bf16  *d_B  = sycl::malloc_device<bf16>(mm_ * mm_, q);
  bf16  *d_BX = sycl::malloc_device<bf16>(N, q);          // mm_ x nn_ = N

  q.memcpy(d_m, mom, size_bytes);
  q.memcpy(d_g, g, size_bytes);
  q.memcpy(d_p, p, size_bytes);
  q.wait();

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int it = 0; it < repeat; it++) {
    // momentum update + bf16 conversion; for P>Q it is fused with the
    // transpose so X is produced directly as the (Q x P) working matrix.
    if (P > Q) {
      q.parallel_for(sycl::range<2>(P, Q), [=](sycl::id<2> id) {
        int i = id[0], j = id[1];
        size_t idx = (size_t)i * Q + j;
        float m = d_m[idx];
        bf16 v = mom_step(m, d_g[idx], beta);
        d_m[idx] = m;
        d_X[(size_t)j * P + i] = v;   // X is Q x P
      });
    } else {
      q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        float m = d_m[i];
        d_X[i] = mom_step(m, d_g[i], beta);
        d_m[i] = m;
      });
    }

    // normalize by Frobenius norm
    q.memset(d_ss, 0, sizeof(float));
    q.submit([&](sycl::handler& h) {
      auto sumr = sycl::reduction(d_ss, sycl::plus<float>());
      h.parallel_for(sycl::range<1>(N), sumr, [=](sycl::id<1> i, auto& sum) {
        float x = (float)d_X[i];
        sum += x * x;
      });
    });
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
      float inv = 1.f / (sycl::sqrt(*d_ss) + 1e-7f);
      d_X[i] = bf16((float)d_X[i] * inv);
    });

    // Newton-Schulz iterations (NS_A, NS_B, and NS_C defined in reference.h)
    for (int s = 0; s < NS_STEPS; s++) {
      // A=X * X^T
      rgemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
            mm_, mm_, nn_, d_X, d_X, d_A);
      // A2=A * A
      rgemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
            mm_, mm_, mm_, d_A, d_A, d_A2);
      // B = b*A + c*A2
      const int nmm = mm_ * mm_;
      q.parallel_for(sycl::range<1>(nmm), [=](sycl::id<1> i) {
        d_B[i] = bf16(NS_B * (float)d_A[i] + NS_C * (float)d_A2[i]);
      });
      // BX=B * X
      rgemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
            mm_, nn_, mm_, d_B, d_X, d_BX);
      // X = a*X + BX
      q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        d_X[i] = bf16(NS_A * (float)d_X[i] + (float)d_BX[i]);
      });
    }

    // spectral scaling + weight update; for P>Q it is fused with the
    // transpose so the (Q x P) working matrix updates p (P x Q) in place.
    if (P > Q) {
      const float decay = 1.f - lr * wd;
      q.parallel_for(sycl::range<2>(P, Q), [=](sycl::id<2> id) {
        int i = id[0], j = id[1];
        size_t k = (size_t)i * Q + j;       // P x Q linear index
        float u = scale * (float)d_X[(size_t)j * P + i];  // X is Q x P
        d_p[k] = d_p[k] * decay - lr * u;
      });
    } else {
      const float decay = 1.f - lr * wd;
      q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        d_p[i] = p_step(d_p[i], (float)d_X[i], decay, lr, scale);
      });
    }
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average step execution time %f (ms)\n", time * 1e-6f / repeat);

  q.memcpy(p, d_p, size_bytes).wait();

  sycl::free(d_p, q);
  sycl::free(d_m, q);
  sycl::free(d_g, q);
  sycl::free(d_ss, q);
  sycl::free(d_X, q);
  sycl::free(d_A, q);
  sycl::free(d_A2, q);
  sycl::free(d_B, q);
  sycl::free(d_BX, q);

  if (verify) {
    reference(repeat, r, rm, g, P, Q, beta, lr, wd);

    bool ok = true;
    for (int i = 0; i < N; i++) {
      if (fabsf(r[i] - p[i]) > 1e-2f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  free(mom);
  free(g);
  free(p);
  free(r);
  free(rm);
  return 0;
}
