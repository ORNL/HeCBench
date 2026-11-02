#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>
#include <hip/hip_runtime.h>
#define HIPBLAS_V2
#include <hipblas/hipblas.h>
#include <rocsolver/rocsolver.h>
#include <string.h>
#include "reference.h"

// One order-2 root-inverse Shampoo step on a single blocked parameter of
// shape (rows x cols). Factor matrices stay in fp32. The inverse fourth root
// is the coupled Newton-Schulz iteration (Polar Express coefficients) and is
// computed with fp32 GEMMs. hipBLAS pedantic math matches cuBLAS pedantic
// math (TF32 off). NewtonSchulzRootInvConfig uses the same products.
//
// https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/preconditioner/shampoo_preconditioner_list.py
//
// The search direction is then rescaled to the Frobenius norm of an AdaGrad
// direction on the same block (Shi et al., arXiv:2309.06497, Section 2.4).
// That is layer-wise learning-rate grafting: Shampoo chooses the direction
// of this block's weight update, and AdaGrad supplies its length.

#define CHECK_HIP(call)                                                       \
  do {                                                                        \
    hipError_t err__ = (call);                                                \
    if (err__ != hipSuccess) {                                                \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,            \
              hipGetErrorString(err__));                                      \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CHECK_HIPBLAS(call)                                                   \
  do {                                                                        \
    hipblasStatus_t st__ = (call);                                            \
    if (st__ != HIPBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "hipBLAS error %d at %s:%d\n", (int)st__, __FILE__,     \
              __LINE__);                                                      \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CHECK_ROCSOLVER(call)                                                  \
  do {                                                                         \
    rocblas_status st__ = (call);                                              \
    if (st__ != rocblas_status_success) {                                     \
      fprintf(stderr, "rocSOLVER error %d at %s:%d\n", (int)st__, __FILE__,   \
              __LINE__);                                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static hipblasHandle_t g_handle;
static rocblas_handle g_solver;
static hipStream_t g_stream = 0;

// 256-thread kernels. Contiguous stages move 128 bits per thread.
// GEMM is full fp32: TF32 makes this Newton-Schulz iteration diverge.
static const int kThreads = 256;

// Row-major C = alpha * op(A) @ op(B) + beta * C. C is M x N and K is the
// inner dimension. hipBLAS is column-major, so the launch is the same product
// on the transposed views: C^T from op(B)^T and op(A)^T.
static void rgemm(hipblasOperation_t opA, hipblasOperation_t opB,
                  int M, int N, int K,
                  const float* A, const float* B, float* C,
                  float alpha, float beta) {
  const int lda = (opA == HIPBLAS_OP_N) ? K : M;
  const int ldb = (opB == HIPBLAS_OP_N) ? N : K;
  const int ldc = N;
  CHECK_HIPBLAS(hipblasSgemm(g_handle, opB, opA, N, M, K,
                             &alpha, B, ldb, A, lda, &beta, C, ldc));
}

// Number of 256-thread blocks that cover `count` floats four at a time.
// A count below 4 still yields one block. Thread 0 of that block handles
// the leftover entries inside the kernels.
static int blocks_for4(int count) {
  int n4 = count >> 2;
  int blocks = (n4 + kThreads - 1) / kThreads;
  return blocks > 0 ? blocks : 1;
}

// Write ||A||_F to *nrm. The sum of squares runs in double, then one square
// root converts it to the norm. One block. Thread t owns float4 slots
// t, t+256, ... so the three ports reduce the same pairs.
// The inverse root divides by this norm, and the ridge is proportional to it.
__global__ void
frobenius_nrm(const float* __restrict__ A, float* __restrict__ nrm, int count) {
  __shared__ double smem[256];
  const int n4 = count >> 2;
  const float4* A4 = reinterpret_cast<const float4*>(A);
  double sum = 0.0;
  for (int i = threadIdx.x; i < n4; i += kThreads) {
    float4 v = A4[i];
    sum += (double)v.x * (double)v.x;
    sum += (double)v.y * (double)v.y;
    sum += (double)v.z * (double)v.z;
    sum += (double)v.w * (double)v.w;
  }
  if (threadIdx.x == 0) {
    for (int j = n4 * 4; j < count; ++j)
      sum += (double)A[j] * (double)A[j];
  }
  smem[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) *nrm = (float)sqrt(smem[0]);
}

// Adds the epsilon-I regularization in the factors of paper Eqs. (14)-(15).
// This implementation uses max(||A||_F * rel_eps, eps), following the cited
// upstream Newton-Schulz solver. A rectangular gradient has a singular Gram
// matrix, so this ridge keeps its inverse root finite.
__global__ void
add_ridge(float* __restrict__ A, const float* __restrict__ nrm,
          float rel_eps, float eps, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) A[(size_t)i * n + i] += fmaxf((*nrm) * rel_eps, eps);
}

// Z_ii = 1. The caller zeros Z first, so Z becomes I, the starting value of
// the inverse-square-root side of the coupled Newton-Schulz iteration.
__global__ void
set_diag(float* __restrict__ Z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) Z[(size_t)i * n + i] = 1.f;
}

__global__ void
scale_eigenvectors(float* Q, const float* w, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = n * n;
  if (i < count) {
    int col = i / n;
    Q[i] *= powf(fmaxf(w[col], kEpsilon), -0.25f);
  }
}

// Bias-correct the EMA factors following paper Section 3.1.1 after
// Eqs. (25)-(26): Lhat=L/(1-beta2^step), and likewise for R.
// The denominator is a host value because each step relaunches the kernels.
__global__ void
scale_div(float* __restrict__ dst, const float* __restrict__ src,
          float denom, int count) {
  const int n4 = count >> 2;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    float4 v = reinterpret_cast<const float4*>(src)[i];
    v.x /= denom; v.y /= denom; v.z /= denom; v.w /= denom;
    reinterpret_cast<float4*>(dst)[i] = v;
  }
  if (i == 0)
    for (int j = n4 * 4; j < count; ++j) dst[j] = src[j] / denom;
}

// Y = a * Y + b * X, entrywise. For factor updates this implements paper
// Eqs. (25)-(26), with a=beta2, b=1-beta2, and X equal to GG^T or G^T G.
// Newton-Schulz also reuses it for its coupled polynomial; that solver is
// from the cited upstream implementation, not an iteration in this paper.
__global__ void
axpy(float* __restrict__ Y, const float* __restrict__ X,
     float a, float b, int count) {
  const int n4 = count >> 2;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    float4 y = reinterpret_cast<float4*>(Y)[i];
    float4 x = reinterpret_cast<const float4*>(X)[i];
    y.x = a * y.x + b * x.x;
    y.y = a * y.y + b * x.y;
    y.z = a * y.z + b * x.z;
    y.w = a * y.w + b * x.w;
    reinterpret_cast<float4*>(Y)[i] = y;
  }
  if (i == 0)
    for (int j = n4 * 4; j < count; ++j) Y[j] = a * Y[j] + b * X[j];
}

// Paper Section 2.2, Eq. (9), diagonal AdaGrad, stored with G's shape:
//   A_t = A_{t-1} + G_t^{odot 2}.
// Section 2.4 and Algorithm 1 use this diagonal state for grafting.
__global__ void
square_accum(float* __restrict__ A, const float* __restrict__ G, int count) {
  const int n4 = count >> 2;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    float4 a = reinterpret_cast<float4*>(A)[i];
    float4 g = reinterpret_cast<const float4*>(G)[i];
    a.x += g.x * g.x;
    a.y += g.y * g.y;
    a.z += g.z * g.z;
    a.w += g.w * g.w;
    reinterpret_cast<float4*>(A)[i] = a;
  }
  if (i == 0)
    for (int j = n4 * 4; j < count; ++j) A[j] += G[j] * G[j];
}

// Paper Section 2.2, Eq. (13), the diagonal AdaGrad direction. Eq. (11) is
// the same componentwise division before epsilon regularization:
//   P_graft = G_t / (sqrt(A_t) + epsilon).
// Section 2.4 and Algorithm 1 use this direction as the graft.
__global__ void
graft_direction(float* __restrict__ dst, const float* __restrict__ G,
                const float* __restrict__ A, float eps, int count) {
  const int n4 = count >> 2;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    float4 g = reinterpret_cast<const float4*>(G)[i];
    float4 a = reinterpret_cast<const float4*>(A)[i];
    g.x /= sqrtf(a.x) + eps;
    g.y /= sqrtf(a.y) + eps;
    g.z /= sqrtf(a.z) + eps;
    g.w /= sqrtf(a.w) + eps;
    reinterpret_cast<float4*>(dst)[i] = g;
  }
  if (i == 0)
    for (int j = n4 * 4; j < count; ++j)
      dst[j] = G[j] / (sqrtf(A[j]) + eps);
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
__global__ void
scale_by_graft(float* __restrict__ Y, const float* __restrict__ nrm_graft,
               const float* __restrict__ nrm_shampoo, int count) {
  const float scale = (*nrm_graft) / ((*nrm_shampoo) + kGraftNormEps);
  const int n4 = count >> 2;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    float4 y = reinterpret_cast<float4*>(Y)[i];
    y.x *= scale; y.y *= scale; y.z *= scale; y.w *= scale;
    reinterpret_cast<float4*>(Y)[i] = y;
  }
  if (i == 0)
    for (int j = n4 * 4; j < count; ++j) Y[j] *= scale;
}

// Y = A / max(*nrm, 1e-8), entrywise. *nrm is ||A||_F.
// Dividing by ||A||_F puts every singular value in [0, 1], the range the
// Polar Express coefficients were fit for. The floor avoids a zero divisor.
__global__ void
div_by_norm(float* __restrict__ Y, const float* __restrict__ A,
            const float* __restrict__ nrm, int count) {
  const float alpha = fmaxf(*nrm, 1e-8f);
  const int n4 = count >> 2;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    float4 v = reinterpret_cast<const float4*>(A)[i];
    v.x /= alpha; v.y /= alpha; v.z /= alpha; v.w /= alpha;
    reinterpret_cast<float4*>(Y)[i] = v;
  }
  if (i == 0)
    for (int j = n4 * 4; j < count; ++j) Y[j] = A[j] / alpha;
}

// Undo the Frobenius normalization after one Newton-Schulz pass.
// s = sqrt(max(*nrm, 1e-8)), and *nrm is ||A||_F from before the division.
// mode 0 writes src * s. Applied to Y, that restores a square root.
// mode 1 writes src / s. Applied to Z, that restores an inverse square root.
__global__ void
scale_from_norm(float* __restrict__ dst, const float* __restrict__ src,
                const float* __restrict__ nrm, int count, int mode) {
  const float alpha = fmaxf(*nrm, 1e-8f);
  const float s = sqrtf(alpha);
  const int n4 = count >> 2;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    float4 v = reinterpret_cast<const float4*>(src)[i];
    if (mode == 0) { v.x *= s; v.y *= s; v.z *= s; v.w *= s; }
    else           { v.x /= s; v.y /= s; v.z /= s; v.w /= s; }
    reinterpret_cast<float4*>(dst)[i] = v;
  }
  if (i == 0) {
    for (int j = n4 * 4; j < count; ++j)
      dst[j] = (mode == 0) ? src[j] * s : src[j] / s;
  }
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
static void inverse_root(float* A, float* Y, float* Z, float* T, float* B,
                         float* nrm, int n) {
  const int count = n * n;
  const int vec = blocks_for4(count);
  const int diag = (n + kThreads - 1) / kThreads;
  frobenius_nrm<<<1, kThreads, 0, g_stream>>>(A, nrm, count);
  add_ridge<<<diag, kThreads, 0, g_stream>>>(A, nrm, kRelativeEpsilon, kEpsilon, n);

  int passes = 0;
  for (int r = kShampooRoot; r > 1; r >>= 1) passes++;
  for (int pass = passes; pass >= 1; pass--) {
    frobenius_nrm<<<1, kThreads, 0, g_stream>>>(A, nrm, count);
    div_by_norm<<<vec, kThreads, 0, g_stream>>>(Y, A, nrm, count);
    CHECK_HIP(hipMemsetAsync(Z, 0, sizeof(float) * (size_t)count, g_stream));
    set_diag<<<diag, kThreads, 0, g_stream>>>(Z, n);
    for (int s = 0; s < kNsIters; s++) {
      float a = kNs[s][0], b = kNs[s][1], c = kNs[s][2];
      rgemm(HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, Z, Y, T, 1.f, 0.f);
      rgemm(HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, T, T, B, c, 0.f);
      axpy<<<vec, kThreads, 0, g_stream>>>(B, T, 1.f, b, count);
      rgemm(HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, Y, B, T, 1.f, 0.f);
      axpy<<<vec, kThreads, 0, g_stream>>>(Y, T, a, 1.f, count);
      rgemm(HIPBLAS_OP_N, HIPBLAS_OP_N, n, n, n, B, Z, T, 1.f, 0.f);
      axpy<<<vec, kThreads, 0, g_stream>>>(Z, T, a, 1.f, count);
    }
    const float* src = (pass > 1) ? Y : Z;
    scale_from_norm<<<vec, kThreads, 0, g_stream>>>(A, src, nrm, count, pass > 1 ? 0 : 1);
  }
}

// rocSOLVER's complete symmetric eigendecomposition is the solver path
// discussed in the cited paper.  The scaled eigenvectors reconstruct A^-1/4.
static void eigh_inverse_root(float* A, float* Q, float* result, float* nrm,
                              float* w, float* offdiag, rocblas_int* info,
                              int n) {
  frobenius_nrm<<<1, kThreads, 0, g_stream>>>(A, nrm, n * n);
  add_ridge<<<(n + kThreads - 1) / kThreads, kThreads, 0, g_stream>>>(
      A, nrm, kRelativeEpsilon, kEpsilon, n);
  CHECK_ROCSOLVER(rocsolver_ssyevd(g_solver, rocblas_evect_original,
                                   rocblas_fill_lower, n, A, n, w, offdiag,
                                   info));
  CHECK_HIP(hipMemcpyAsync(Q, A, sizeof(float) * (size_t)n * n,
                           hipMemcpyDeviceToDevice, g_stream));
  scale_eigenvectors<<<(n * n + kThreads - 1) / kThreads, kThreads, 0,
                       g_stream>>>(A, w, n);
  rgemm(HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, n, A, Q, result, 1.f, 0.f);
  CHECK_HIP(hipMemcpyAsync(A, result, sizeof(float) * (size_t)n * n,
                           hipMemcpyDeviceToDevice, g_stream));
}

// Time `steps` Shampoo updates of one frozen M x N gradient. verify != 0
// compares the preconditioned gradient with the host reference.
int main(int argc, char* argv[]) {
  if (argc != 5 && argc != 6) {
    printf("Usage: %s <rows> <cols> <steps> <verify> [newton-schulz|eigendecomposition|both]\n", argv[0]);
    printf("  rows x cols is one order-2 Shampoo block\n");
    return 1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int steps = atoi(argv[3]);
  const int verify = atoi(argv[4]);
  const char* method = argc == 6 ? argv[5] : "both";
  const bool run_ns = strcmp(method, "newton-schulz") == 0 || strcmp(method, "both") == 0;
  const bool run_eigh = strcmp(method, "eigendecomposition") == 0 || strcmp(method, "both") == 0;
  if (M < 1 || N < 1 || M > 8192 || N > 8192 || steps < 1) {
    fprintf(stderr, "rows and cols must be in [1, 8192], steps >= 1\n");
    return 1;
  }
  if (!run_ns && !run_eigh) {
    fprintf(stderr, "method must be newton-schulz, eigendecomposition, or both\n");
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

  float *d_G, *d_L, *d_R, *d_Linv, *d_Rinv, *d_out, *d_A;
  float *d_Y, *d_Z, *d_T, *d_B, *d_nrm, *d_eigenvalues, *d_offdiag;
  rocblas_int* d_solver_info;
  CHECK_HIP(hipMalloc(&d_G, sizeof(float) * (size_t)gcount));
  CHECK_HIP(hipMalloc(&d_L, sizeof(float) * (size_t)lcount));
  CHECK_HIP(hipMalloc(&d_R, sizeof(float) * (size_t)rcount));
  CHECK_HIP(hipMalloc(&d_Linv, sizeof(float) * (size_t)lcount));
  CHECK_HIP(hipMalloc(&d_Rinv, sizeof(float) * (size_t)rcount));
  CHECK_HIP(hipMalloc(&d_out, sizeof(float) * (size_t)gcount));
  CHECK_HIP(hipMalloc(&d_A, sizeof(float) * (size_t)gcount));
  CHECK_HIP(hipMalloc(&d_Y, sizeof(float) * (size_t)nmax * nmax));
  CHECK_HIP(hipMalloc(&d_Z, sizeof(float) * (size_t)nmax * nmax));
  CHECK_HIP(hipMalloc(&d_T, sizeof(float) * (size_t)nmax * nmax));
  CHECK_HIP(hipMalloc(&d_B, sizeof(float) * (size_t)nmax * nmax));
  // Two scalars: ||P_shampoo||_F and ||P_graft||_F. The inverse root uses
  // the first slot and finishes before grafting writes either one.
  CHECK_HIP(hipMalloc(&d_nrm, sizeof(float) * 2));
  CHECK_HIP(hipMalloc(&d_eigenvalues, sizeof(float) * nmax));
  CHECK_HIP(hipMalloc(&d_offdiag, sizeof(float) * std::max(1, nmax - 1)));
  // Verification keeps one status per factor solve. Timing reuses one slot
  // and does not read it.
  const int solver_info_count = (verify && run_eigh) ? 2 * steps : 1;
  CHECK_HIP(hipMalloc(&d_solver_info, sizeof(rocblas_int) * (size_t)solver_info_count));

  CHECK_HIP(hipMemcpy(d_G, G, sizeof(float) * (size_t)gcount, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemset(d_L, 0, sizeof(float) * (size_t)lcount));
  CHECK_HIP(hipMemset(d_R, 0, sizeof(float) * (size_t)rcount));
  CHECK_HIP(hipMemset(d_A, 0, sizeof(float) * (size_t)gcount));

  CHECK_HIPBLAS(hipblasCreate(&g_handle));
  // Same full-fp32 request as cublasSetMathMode(CUBLAS_PEDANTIC_MATH).
  // rocBLAS does not implement pedantic mode and returns NOT_SUPPORTED.
  // Its default SGEMM is already full fp32; xf32 is HIPBLAS_XF32_XDL_MATH,
  // which this handle does not select. SYCL requests oneMKL standard.
  hipblasStatus_t math_st = hipblasSetMathMode(g_handle, HIPBLAS_PEDANTIC_MATH);
  if (math_st == HIPBLAS_STATUS_NOT_SUPPORTED)
    math_st = hipblasSetMathMode(g_handle, HIPBLAS_DEFAULT_MATH);
  if (math_st != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "hipBLAS error %d at %s:%d\n", (int)math_st, __FILE__,
            __LINE__);
    exit(EXIT_FAILURE);
  }
  CHECK_HIP(hipStreamCreate(&g_stream));
  CHECK_HIPBLAS(hipblasSetStream(g_handle, g_stream));
  CHECK_ROCSOLVER(rocblas_create_handle(&g_solver));
  CHECK_ROCSOLVER(rocblas_set_stream(g_solver, g_stream));

  // One step on d_G. Verification replaces d_G each step; timing leaves it
  // fixed. L and R are the row and column factors. Launched directly each
  // step, as on CUDA and SYCL. bc is the host bias
  // correction 1 - beta2^step.
  //   Eqs. (25)-(26): EMA updates of L from GG^T and R from G^T G
  //   Section 3.1.1: bias correction, then Algorithm 1: inverse fourth roots
  //   Eq. (17): out = Linv^T G Rinv (Linv is symmetric)
  //   Section 2.2, Eqs. (9) and (13): diagonal AdaGrad state and direction
  //   Eq. (20): rescale out by ||P_graft||_F / ||out||_F
  auto run_step = [&](float bc, float* save_pre, bool use_eigh,
                      rocblas_int* info_l, rocblas_int* info_r) {
    rgemm(HIPBLAS_OP_N, HIPBLAS_OP_T, M, M, N, d_G, d_G, d_T, 1.f, 0.f);
    axpy<<<blocks_for4(lcount), kThreads, 0, g_stream>>>(d_L, d_T, kBeta2, kWeight, lcount);
    rgemm(HIPBLAS_OP_T, HIPBLAS_OP_N, N, N, M, d_G, d_G, d_T, 1.f, 0.f);
    axpy<<<blocks_for4(rcount), kThreads, 0, g_stream>>>(d_R, d_T, kBeta2, kWeight, rcount);

    scale_div<<<blocks_for4(lcount), kThreads, 0, g_stream>>>(d_Linv, d_L, bc, lcount);
    if (use_eigh)
      eigh_inverse_root(d_Linv, d_Y, d_Z, d_nrm, d_eigenvalues, d_offdiag,
                        info_l, M);
    else
      inverse_root(d_Linv, d_Y, d_Z, d_T, d_B, d_nrm, M);
    scale_div<<<blocks_for4(rcount), kThreads, 0, g_stream>>>(d_Rinv, d_R, bc, rcount);
    if (use_eigh)
      eigh_inverse_root(d_Rinv, d_Y, d_Z, d_nrm, d_eigenvalues, d_offdiag,
                        info_r, N);
    else
      inverse_root(d_Rinv, d_Y, d_Z, d_T, d_B, d_nrm, N);

    rgemm(HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, M, d_Linv, d_G, d_T, 1.f, 0.f);
    rgemm(HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, N, d_T, d_Rinv, d_out, 1.f, 0.f);
    if (save_pre)
      CHECK_HIP(hipMemcpyAsync(save_pre, d_out, sizeof(float) * (size_t)gcount,
                               hipMemcpyDeviceToHost, g_stream));

    square_accum<<<blocks_for4(gcount), kThreads, 0, g_stream>>>(d_A, d_G, gcount);
    graft_direction<<<blocks_for4(gcount), kThreads, 0, g_stream>>>(
        d_T, d_G, d_A, kGraftEpsilon, gcount);
    frobenius_nrm<<<1, kThreads, 0, g_stream>>>(d_out, d_nrm, gcount);
    frobenius_nrm<<<1, kThreads, 0, g_stream>>>(d_T, d_nrm + 1, gcount);
    scale_by_graft<<<blocks_for4(gcount), kThreads, 0, g_stream>>>(
        d_out, d_nrm + 1, d_nrm, gcount);
  };

  int rc = 0;
  const char* names[] = {"newton-schulz", "eigendecomposition"};
  const bool selected[] = {run_ns, run_eigh};
  for (int mi = 0; mi < 2; ++mi) {
    if (!selected[mi]) continue;
    const bool use_eigh = mi == 1;
    CHECK_HIP(hipMemcpyAsync(d_G, G, sizeof(float) * (size_t)gcount,
                             hipMemcpyHostToDevice, g_stream));
    CHECK_HIP(hipMemsetAsync(d_L, 0, sizeof(float) * (size_t)lcount, g_stream));
    CHECK_HIP(hipMemsetAsync(d_R, 0, sizeof(float) * (size_t)rcount, g_stream));
    CHECK_HIP(hipMemsetAsync(d_A, 0, sizeof(float) * (size_t)gcount, g_stream));
    if (verify && use_eigh)
      CHECK_HIP(hipMemsetAsync(d_solver_info, 0,
                               sizeof(rocblas_int) * (size_t)solver_info_count,
                               g_stream));
    CHECK_HIP(hipStreamSynchronize(g_stream));
    auto start = std::chrono::steady_clock::now();
    for (int step = 1; step <= steps; step++) {
      if (verify) {
        shampoo_gradient_at_step(G, H, Gs, gcount, step);
        CHECK_HIP(hipMemcpyAsync(d_G, Gs, sizeof(float) * (size_t)gcount,
                                 hipMemcpyHostToDevice, g_stream));
      }
      rocblas_int* info_l =
          d_solver_info + ((verify && use_eigh) ? (step - 1) * 2 : 0);
      run_step(shampoo_bias_correction2(step),
               (verify && step == steps) ? pre : nullptr, use_eigh,
               info_l, info_l + ((verify && use_eigh) ? 1 : 0));
    }
    CHECK_HIP(hipStreamSynchronize(g_stream));
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("rows %d cols %d steps %d method %s\n", M, N, steps, names[mi]);
    printf("Average shampoo step time (%s) %f (ms)\n", names[mi], ms / steps);
    if (verify && use_eigh) {
      rocblas_int* solver_info = (rocblas_int*)malloc(
          sizeof(rocblas_int) * (size_t)solver_info_count);
      if (!solver_info) {
        fprintf(stderr, "host allocation failed\n");
        rc = 1;
        break;
      }
      CHECK_HIP(hipMemcpy(solver_info, d_solver_info,
                          sizeof(rocblas_int) * (size_t)solver_info_count,
                          hipMemcpyDeviceToHost));
      int bad = 0;
      for (int i = 0; i < solver_info_count; ++i) {
        if (solver_info[i] != 0) {
          bad = (int)solver_info[i];
          break;
        }
      }
      free(solver_info);
      if (bad != 0) {
        fprintf(stderr, "rocSOLVER syevd did not converge (info=%d)\n", bad);
        printf("%s FAIL\n", names[mi]);
        rc = 1;
        break;
      }
    }
    CHECK_HIP(hipMemcpy(out, d_out, sizeof(float) * (size_t)gcount, hipMemcpyDeviceToHost));
    if (verify) {
      CHECK_HIP(hipMemcpy(Lh, d_L, sizeof(float) * (size_t)lcount, hipMemcpyDeviceToHost));
      CHECK_HIP(hipMemcpy(Rh, d_R, sizeof(float) * (size_t)rcount, hipMemcpyDeviceToHost));
      int ok = ref_shampoo(G, H, ref_L, ref_R, ref_pre, ref, M, N, steps) == 0 &&
               shampoo_match("L", Lh, ref_L, lcount) &&
               shampoo_match("R", Rh, ref_R, rcount) &&
               shampoo_match("pre-graft", pre, ref_pre, gcount) &&
               shampoo_match("grafted", out, ref, gcount);
      printf("%s %s\n", names[mi], ok ? "PASS" : "FAIL");
      if (!ok) {
        rc = 1;
        break;
      }
    }
  }
  CHECK_HIPBLAS(hipblasDestroy(g_handle));
  CHECK_ROCSOLVER(rocblas_destroy_handle(g_solver));
  CHECK_HIP(hipFree(d_G));
  CHECK_HIP(hipFree(d_L));
  CHECK_HIP(hipFree(d_R));
  CHECK_HIP(hipFree(d_Linv));
  CHECK_HIP(hipFree(d_Rinv));
  CHECK_HIP(hipFree(d_out));
  CHECK_HIP(hipFree(d_A));
  CHECK_HIP(hipFree(d_Y));
  CHECK_HIP(hipFree(d_Z));
  CHECK_HIP(hipFree(d_T));
  CHECK_HIP(hipFree(d_B));
  CHECK_HIP(hipFree(d_nrm));
  CHECK_HIP(hipFree(d_eigenvalues));
  CHECK_HIP(hipFree(d_offdiag));
  CHECK_HIP(hipFree(d_solver_info));
  CHECK_HIP(hipStreamDestroy(g_stream));

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
