#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <chrono>
#include <random>
#include <algorithm>
#include "reference.h"

// The Muon optimizer is applied to a single 2D hidden weight matrix of shape
// (rows x cols) == (out_features x in_features) of a Linear layer. The momentum
// buffer, the gradient and the weights are kept in float32; the Newton-Schulz
// orthogonalization runs in bfloat16 (with float32 accumulation), as in
// the reference implementation https://github.com/KellerJordan/Muon .
//
// This version uses cuBLAS bf16 GEMMs with fp32 accumulation for
// the Newton-Schulz matmuls, and fuses the elementwise steps:
//   * momentum update + fp32->bf16 conversion  -> momentum_update
//   * Newton-Schulz spectral scaling            -> folded into param_update

typedef __nv_bfloat16 bf16;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err__));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t st__ = (call);                                             \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", st__, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// 8-byte (4 x bf16) vector for 128-bit access
struct __align__(8) bf16x4 { __nv_bfloat162 lo, hi; };

static cublasHandle_t g_handle;

// row-major C(M x N) = opA(A) @ opB(B), bf16 storage, fp32 accumulation.
// cuBLAS is column-major, so we compute C^T = opB(B)^T @ opA(A)^T by swapping
// the operands; a row-major (r x c) array is a column-major (c x r) array.
static inline void rgemm(cublasOperation_t opA, cublasOperation_t opB,
                         int M, int N, int K,
                         const bf16* A, const bf16* B, bf16* C)
{
  const float alpha = 1.f, beta = 0.f;
  const int lda = (opA == CUBLAS_OP_N) ? K : M;
  const int ldb = (opB == CUBLAS_OP_N) ? N : K;
  const int ldc = N;
  CHECK_CUBLAS(cublasGemmEx(g_handle, opB, opA, N, M, K,
               &alpha,
               B, CUDA_R_16BF, ldb,
               A, CUDA_R_16BF, lda,
               &beta,
               C, CUDA_R_16BF, ldc,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

//   mom = beta*mom + (1-beta)*g; out = bf16((1-beta)*g + beta*mom)
__device__ __forceinline__ bf16 mom_step(float& m, float g, float beta)
{
  float t = (1.f - beta) * g;
  float mi = beta * m + t;
  m = mi;
  return __float2bfloat16(t + beta * mi);
}

// Vectorized: 4 elements/thread via float4 loads and a bf16x4 store.
__global__ void momentum_update(float* __restrict__ mom,
                                const float* __restrict__ g,
                                bf16* __restrict__ out,
                                float beta, int n)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = n >> 2;
  if (idx < n4) {
    float4 m  = reinterpret_cast<float4*>(mom)[idx];
    float4 gg = reinterpret_cast<const float4*>(g)[idx];
    bf16x4 o;
    o.lo.x = mom_step(m.x, gg.x, beta);
    o.lo.y = mom_step(m.y, gg.y, beta);
    o.hi.x = mom_step(m.z, gg.z, beta);
    o.hi.y = mom_step(m.w, gg.w, beta);
    reinterpret_cast<float4*>(mom)[idx] = m;
    reinterpret_cast<bf16x4*>(out)[idx] = o;
  }
  if (idx == 0)          // scalar tail (n not a multiple of 4)
    for (int i = n4 * 4; i < n; i++)
      out[i] = mom_step(mom[i], g[i], beta);
}

// For the P>Q ("tall") case the two
// transposes are fused into the momentum update and the param update below, so
// there is no separate transpose pass or intermediate buffer.
#define T_TILE  32
#define T_BROWS 8
#define SWIZZLE(row, col) ((col) ^ (row))

// momentum update (fp32 state, P x Q) written transposed as the
// bf16 Newton-Schulz working matrix X (Q x P).  Replaces momentum_update +
// a standalone transpose, avoiding the intermediate (P x Q) bf16 buffer.
__global__ void momentum_transpose(float* __restrict__ mom,
                                   const float* __restrict__ g,
                                   bf16* __restrict__ X,
                                   float beta, int P, int Q)
{
  __shared__ bf16 tile[T_TILE][T_TILE];

  int x = blockIdx.x * T_TILE + threadIdx.x;   // input column [0,Q)
  int y = blockIdx.y * T_TILE + threadIdx.y;   // input row    [0,P)
#pragma unroll
  for (int j = 0; j < T_TILE; j += T_BROWS)
    if (x < Q && (y + j) < P) {
      size_t idx = (size_t)(y + j) * Q + x;
      float m = mom[idx];
      tile[threadIdx.y + j][SWIZZLE(threadIdx.y + j, threadIdx.x)] =
          mom_step(m, g[idx], beta);
      mom[idx] = m;
    }

  __syncthreads();

  int xo = blockIdx.y * T_TILE + threadIdx.x;  // output column [0,P)
  int yo = blockIdx.x * T_TILE + threadIdx.y;  // output row    [0,Q)
#pragma unroll
  for (int j = 0; j < T_TILE; j += T_BROWS)
    if (xo < P && (yo + j) < Q)
      X[(size_t)(yo + j) * P + xo] =
          tile[threadIdx.x][SWIZZLE(threadIdx.x, threadIdx.y + j)];
}

// The (Q x P) working matrix X is transposed back to the (P x Q)
// update and the spectral-scaling + weight update is applied to p in place.
// Replaces a standalone transpose + param_update, avoiding the intermediate
// (P x Q) bf16 buffer.
__global__ void transpose_param_update(float* __restrict__ p,
                                       const bf16* __restrict__ X,
                                       float lr, float wd, float scale,
                                       int P, int Q)
{
  __shared__ bf16 tile[T_TILE][T_TILE];

  int x = blockIdx.x * T_TILE + threadIdx.x;   // X column [0,P)
  int y = blockIdx.y * T_TILE + threadIdx.y;   // X row    [0,Q)
#pragma unroll
  for (int j = 0; j < T_TILE; j += T_BROWS)
    if (x < P && (y + j) < Q)
      tile[threadIdx.y + j][SWIZZLE(threadIdx.y + j, threadIdx.x)] =
          X[(size_t)(y + j) * P + x];

  __syncthreads();

  const float decay = 1.f - lr * wd;
  int xo = blockIdx.y * T_TILE + threadIdx.x;  // p column [0,Q)
  int yo = blockIdx.x * T_TILE + threadIdx.y;  // p row    [0,P)
#pragma unroll
  for (int j = 0; j < T_TILE; j += T_BROWS)
    if (xo < Q && (yo + j) < P) {
      size_t k = (size_t)(yo + j) * Q + xo;    // P x Q linear index
      float u = scale * __bfloat162float(
          tile[threadIdx.x][SWIZZLE(threadIdx.x, threadIdx.y + j)]);
      p[k] = p[k] * decay - lr * u;
    }
}

// sum of squares of a bf16 array (accumulated in fp32), vectorized bf16x4 loads
__global__ void sumsq_kernel(const bf16* __restrict__ X,
                             float* __restrict__ out, int n)
{
  __shared__ float sdata[256];
  const int tid = threadIdx.x;
  const int n4 = n >> 2;
  float v = 0.f;
  for (int j = blockIdx.x * blockDim.x + tid; j < n4;
       j += blockDim.x * gridDim.x) {
    bf16x4 x = reinterpret_cast<const bf16x4*>(X)[j];
    float2 a = __bfloat1622float2(x.lo);
    float2 b = __bfloat1622float2(x.hi);
    v += a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y;
  }
  if (blockIdx.x == 0 && tid == 0)   // scalar tail
    for (int i = n4 * 4; i < n; i++) {
      float x = __bfloat162float(X[i]);
      v += x * x;
    }
  sdata[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, sdata[0]);
}

// X <- X / (||X||_F + eps),  sumsq holds ||X||_F^2 (from sumsq_kernel)
__global__ void normalize(bf16* __restrict__ X,
                          const float* __restrict__ sumsq, int n)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = n >> 2;
  const float inv = 1.f / (sqrtf(*sumsq) + 1e-7f);
  if (idx < n4) {
    bf16x4 x = reinterpret_cast<bf16x4*>(X)[idx];
    float2 a = __bfloat1622float2(x.lo);
    float2 b = __bfloat1622float2(x.hi);
    a.x *= inv; a.y *= inv; b.x *= inv; b.y *= inv;
    x.lo = __float22bfloat162_rn(a);
    x.hi = __float22bfloat162_rn(b);
    reinterpret_cast<bf16x4*>(X)[idx] = x;
  }
  if (idx == 0)
    for (int i = n4 * 4; i < n; i++)
      X[i] = __float2bfloat16(__bfloat162float(X[i]) * inv);
}

// B = b*A + c*A2
__global__ void combineB(const bf16* __restrict__ A,
                         const bf16* __restrict__ A2,
                         bf16* __restrict__ B,
                         float b, float c, int n)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = n >> 2;
  if (idx < n4) {
    bf16x4 va  = reinterpret_cast<const bf16x4*>(A)[idx];
    bf16x4 va2 = reinterpret_cast<const bf16x4*>(A2)[idx];
    float2 a_lo = __bfloat1622float2(va.lo),  a_hi = __bfloat1622float2(va.hi);
    float2 b_lo = __bfloat1622float2(va2.lo), b_hi = __bfloat1622float2(va2.hi);
    bf16x4 o;
    o.lo = __float22bfloat162_rn(make_float2(b * a_lo.x + c * b_lo.x,
                                             b * a_lo.y + c * b_lo.y));
    o.hi = __float22bfloat162_rn(make_float2(b * a_hi.x + c * b_hi.x,
                                             b * a_hi.y + c * b_hi.y));
    reinterpret_cast<bf16x4*>(B)[idx] = o;
  }
  if (idx == 0)
    for (int i = n4 * 4; i < n; i++)
      B[i] = __float2bfloat16(b * __bfloat162float(A[i]) +
                              c * __bfloat162float(A2[i]));
}

// X = a*X + BX
__global__ void updateX(bf16* __restrict__ X,
                        const bf16* __restrict__ BX,
                        float a, int n)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = n >> 2;
  if (idx < n4) {
    bf16x4 vx = reinterpret_cast<bf16x4*>(X)[idx];
    bf16x4 vb = reinterpret_cast<const bf16x4*>(BX)[idx];
    float2 x_lo = __bfloat1622float2(vx.lo), x_hi = __bfloat1622float2(vx.hi);
    float2 b_lo = __bfloat1622float2(vb.lo), b_hi = __bfloat1622float2(vb.hi);
    vx.lo = __float22bfloat162_rn(make_float2(a * x_lo.x + b_lo.x,
                                              a * x_lo.y + b_lo.y));
    vx.hi = __float22bfloat162_rn(make_float2(a * x_hi.x + b_hi.x,
                                              a * x_hi.y + b_hi.y));
    reinterpret_cast<bf16x4*>(X)[idx] = vx;
  }
  if (idx == 0)
    for (int i = n4 * 4; i < n; i++)
      X[i] = __float2bfloat16(a * __bfloat162float(X[i]) +
                              __bfloat162float(BX[i]));
}

//        p = p*(1 - lr*wd) - lr*update
__device__ __forceinline__ float p_step(float p, float upd,
                                         float decay, float lr, float scale)
{
  return p * decay - lr * (scale * upd);
}

__global__ void param_update(float* __restrict__ p,
                             const bf16* __restrict__ update,
                             float lr, float wd, float scale, int n)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n4 = n >> 2;
  const float decay = 1.f - lr * wd;
  if (idx < n4) {
    float4 vp = reinterpret_cast<float4*>(p)[idx];
    bf16x4 vu = reinterpret_cast<const bf16x4*>(update)[idx];
    float2 u_lo = __bfloat1622float2(vu.lo), u_hi = __bfloat1622float2(vu.hi);
    vp.x = p_step(vp.x, u_lo.x, decay, lr, scale);
    vp.y = p_step(vp.y, u_lo.y, decay, lr, scale);
    vp.z = p_step(vp.z, u_hi.x, decay, lr, scale);
    vp.w = p_step(vp.w, u_hi.y, decay, lr, scale);
    reinterpret_cast<float4*>(p)[idx] = vp;
  }
  if (idx == 0)
    for (int i = n4 * 4; i < n; i++)
      p[i] = p_step(p[i], __bfloat162float(update[i]), decay, lr, scale);
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

  float *d_p, *d_m, *d_g, *d_ss;
  bf16  *d_X, *d_A, *d_A2, *d_B, *d_BX;
  CHECK_CUDA(cudaMalloc((void**)&d_p, size_bytes));
  CHECK_CUDA(cudaMalloc((void**)&d_m, size_bytes));
  CHECK_CUDA(cudaMalloc((void**)&d_g, size_bytes));
  CHECK_CUDA(cudaMalloc((void**)&d_ss, sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&d_X,  sizeof(bf16) * N));          // mm_ x nn_ = N
  CHECK_CUDA(cudaMalloc((void**)&d_A,  sizeof(bf16) * mm_ * mm_));
  CHECK_CUDA(cudaMalloc((void**)&d_A2, sizeof(bf16) * mm_ * mm_));
  CHECK_CUDA(cudaMalloc((void**)&d_B,  sizeof(bf16) * mm_ * mm_));
  CHECK_CUDA(cudaMalloc((void**)&d_BX, sizeof(bf16) * N));          // mm_ x nn_ = N

  CHECK_CUDA(cudaMemcpy(d_m, mom, size_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_g, g, size_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_p, p, size_bytes, cudaMemcpyHostToDevice));

  CHECK_CUBLAS(cublasCreate(&g_handle));

  const int tpb = 256;
  // vectorized (4 elem/thread) launch grids for the elementwise kernels
  const dim3 grid_N4 (((N + 3) / 4 + tpb - 1) / tpb);
  const dim3 grid_mm4 (((mm_ * mm_ + 3) / 4 + tpb - 1) / tpb);
  const dim3 block1 (tpb);
  const dim3 tblock2 (T_TILE, T_BROWS);   // tiled transpose block

  // launch grids for the fused transposing kernels (P>Q path)
  const dim3 gt_fwd ((Q + T_TILE - 1) / T_TILE, (P + T_TILE - 1) / T_TILE);
  const dim3 gt_bwd ((P + T_TILE - 1) / T_TILE, (Q + T_TILE - 1) / T_TILE);

  CHECK_CUDA(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  for (int it = 0; it < repeat; it++) {
    // momentum update + bf16 conversion; for P>Q it is fused with the
    // transpose so X is produced directly as the (Q x P) working matrix.
    if (P > Q)
      momentum_transpose<<<gt_fwd, tblock2>>>(d_m, d_g, d_X, beta, P, Q);
    else
      momentum_update<<<grid_N4, block1>>>(d_m, d_g, d_X, beta, N);

    // normalize by Frobenius norm
    CHECK_CUDA(cudaMemset(d_ss, 0, sizeof(float)));
    sumsq_kernel<<<256, block1>>>(d_X, d_ss, N);
    normalize<<<grid_N4, block1>>>(d_X, d_ss, N);

    // Newton-Schulz iterations (NS_A, NS_B, and NS_C defined in reference.h)
    for (int s = 0; s < NS_STEPS; s++) {
      // A=X * X^T
      rgemm(CUBLAS_OP_N, CUBLAS_OP_T, mm_, mm_, nn_, d_X, d_X, d_A);
      // A2=A * A
      rgemm(CUBLAS_OP_N, CUBLAS_OP_N, mm_, mm_, mm_, d_A, d_A, d_A2);
      // B = b*A + c*A2
      combineB<<<grid_mm4, block1>>>(d_A, d_A2, d_B, NS_B, NS_C, mm_ * mm_);
      // BX=B * X
      rgemm(CUBLAS_OP_N, CUBLAS_OP_N, mm_, nn_, mm_, d_B, d_X, d_BX);
      // X = a*X + BX
      updateX<<<grid_N4, block1>>>(d_X, d_BX, NS_A, N);
    }

    // spectral scaling + weight update; for P>Q it is fused with the
    // transpose so the (Q x P) working matrix updates p (P x Q) in place.
    if (P > Q)
      transpose_param_update<<<gt_bwd, tblock2>>>(d_p, d_X, lr, wd, scale, P, Q);
    else
      param_update<<<grid_N4, block1>>>(d_p, d_X, lr, wd, scale, N);
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average step execution time %f (ms)\n", time * 1e-6f / repeat);

  CHECK_CUDA(cudaMemcpy(p, d_p, size_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUBLAS(cublasDestroy(g_handle));
  CHECK_CUDA(cudaFree(d_p));
  CHECK_CUDA(cudaFree(d_m));
  CHECK_CUDA(cudaFree(d_g));
  CHECK_CUDA(cudaFree(d_ss));
  CHECK_CUDA(cudaFree(d_X));
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_A2));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_BX));

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
