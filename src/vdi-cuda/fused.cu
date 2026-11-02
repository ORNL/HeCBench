// One block per matrix, one launch. Shared memory holds the lower Cholesky
// factor of I+A and then its inverse, plus a 64x64 scratch for the block
// couplings. The inverse is written to global T, B is loaded over the factor,
// and J = B * inv is written to global memory. T is then scaled by alpha.
//
// d = 128 is fixed. When the device has 82432 bytes, the row stride in shared
// memory is 129, not 128: 129 % 32 is 1, so consecutive rows sit in consecutive
// banks. A stride of 128 makes every column read a 32-way conflict. Global
// matrices stay tightly packed; only the shared copy is padded. float4 loads
// come from that packed layout. A 64KB device keeps stride 128 and puts the
// 64x64 scratch in global memory. local_layout.hpp is that choice for HIP and
// SYCL as well.

#include "fused.h"
#include "local_layout.hpp"

#include <cstdio>

constexpr int kDim = 128;
constexpr int kThreads = 256;
constexpr int kNb = 32;

// Right-looking lower Cholesky, in place. 256 threads, two per row: tid and
// tid+128 share row (tid & 127). Only the first of the pair scales the pivot
// column. Each then owns half of the columns (k, row] and applies a 4-wide
// rank-1 update. The loops stay unroll-1; unrolling this divergent per-row
// trip count inflates the critical path without hiding the shared-memory
// latency. The diagonal sqrt and the column scale are the cross-thread
// dependencies, so each of those steps is followed by a block barrier.
__device__ void chol_lower(float* L, int ld, int n, int tid, int* pivot_fail) {
  const int row = tid & (n - 1);
  const int part = tid >> 7;
  float* rp = L + row * ld;
  for (int k = 0; k < n; ++k) {
    if (tid == k) {
      const float d = L[k * ld + k];
      // A non-positive pivot is a failed factorization. sqrt would turn it
      // into NaN and the relative-error check would be the only report.
      if (!(d > 0.0f)) atomicExch(pivot_fail, 1);
      L[k * ld + k] = sqrtf(d);
    }
    __syncthreads();
    const float invd = 1.0f / L[k * ld + k];
    if (part == 0 && row > k) rp[k] *= invd;
    __syncthreads();
    if (row > k) {
      const float lik = rp[k];
      const int lo = k + 1;
      const int hi = row;
      const int mid = (lo + hi) >> 1;
      const int j0 = part ? mid + 1 : lo;
      const int j1 = part ? hi : mid;
      int j = j0;
      #pragma unroll 1
      for (; j + 3 <= j1; j += 4) {
        const float b0 = L[(j + 0) * ld + k];
        const float b1 = L[(j + 1) * ld + k];
        const float b2 = L[(j + 2) * ld + k];
        const float b3 = L[(j + 3) * ld + k];
        rp[j] -= lik * b0;
        rp[j + 1] -= lik * b1;
        rp[j + 2] -= lik * b2;
        rp[j + 3] -= lik * b3;
      }
      #pragma unroll 1
      for (; j <= j1; ++j) rp[j] -= lik * L[j * ld + k];
    }
    __syncthreads();
  }
}

// In-place inverse of one 32x32 lower-triangular block. Lane `lane` owns
// column `lane`: it loads that column into registers, solves L x = e_lane,
// and writes the column back. Entries above the diagonal stay zero, so the
// sum for row i only includes columns j with lane <= j < i. L[i, j] is
// broadcast from the lane that owns column j. No barrier inside the 32.
__device__ void warp_trinv32(float* S, int ld, int r0, int lane) {
  float colv[kNb];
  #pragma unroll
  for (int i = 0; i < kNb; ++i)
    colv[i] = S[(r0 + i) * ld + (r0 + lane)];

  float x[kNb];
  #pragma unroll
  for (int i = 0; i < kNb; ++i) x[i] = 0.0f;
  x[lane] = 1.0f / colv[lane];

  #pragma unroll
  for (int i = 1; i < kNb; ++i) {
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < i; ++j) {
      const float Lij = __shfl_sync(0xffffffff, colv[i], j);
      if (j >= lane) sum += Lij * x[j];
    }
    const float Lii = __shfl_sync(0xffffffff, colv[i], i);
    if (i > lane) x[i] = -sum / Lii;
  }

  #pragma unroll
  for (int i = lane; i < kNb; ++i)
    S[(r0 + i) * ld + (r0 + lane)] = x[i];
}

// C = scale * A * B, row-major, one output row-tile of 8 columns per task.
// The K loop is a single fp32 chain. Callers use this only for the 32 and 64
// couplings of the triangular inverse, where K is short enough for that chain.
__device__ void gemm_nn(const float* A, int lda, const float* B, int ldb,
                        float* C, int ldc, int M, int N, int K,
                        int tid, float scale) {
  constexpr int W = 8;
  const int chunks = N / W;
  const int ntasks = M * chunks;
  for (int task = tid; task < ntasks; task += kThreads) {
    const int row = task / chunks;
    const int c0 = (task - row * chunks) * W;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    const float* arow = A + row * lda;
    for (int k = 0; k < K; ++k) {
      const float aik = arow[k];
      const float* bk = B + k * ldb + c0;
      a0 += aik * bk[0];
      a1 += aik * bk[1];
      a2 += aik * bk[2];
      a3 += aik * bk[3];
      a4 += aik * bk[4];
      a5 += aik * bk[5];
      a6 += aik * bk[6];
      a7 += aik * bk[7];
    }
    float* crow = C + row * ldc + c0;
    crow[0] = scale * a0;
    crow[1] = scale * a1;
    crow[2] = scale * a2;
    crow[3] = scale * a3;
    crow[4] = scale * a4;
    crow[5] = scale * a5;
    crow[6] = scale * a6;
    crow[7] = scale * a7;
  }
}

// C = X^T X. Eight warps, and each walks two 8-column panels. A lane owns
// rows lane, lane+32, lane+64, and lane+96. In one instruction the warp's
// row reads are X[k, lane + i*32]; with stride 129 those 32 addresses hit
// 32 different banks. The eight panel values are the same address for every
// lane, so they broadcast. One accumulator is enough for inv = Linv^T Linv:
// the upper triangle of Linv is zero, so a full k=0..127 dot matches the
// triangular dot that would start at max(row, col).
__device__ void gemm_tn_full(const float* X, int ldx, float* C, int ldc,
                             int n, int tid) {
  constexpr int W = 8;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int nwarps = kThreads >> 5;
  const int panels = n / W;
  for (int p = warp; p < panels; p += nwarps) {
    const int c0 = p * W;
    float acc[4][W] = {};
    for (int k = 0; k < n; ++k) {
      float b0, b1, b2, b3, b4, b5, b6, b7;
      b0 = X[k * ldx + c0 + 0];
      b1 = X[k * ldx + c0 + 1];
      b2 = X[k * ldx + c0 + 2];
      b3 = X[k * ldx + c0 + 3];
      b4 = X[k * ldx + c0 + 4];
      b5 = X[k * ldx + c0 + 5];
      b6 = X[k * ldx + c0 + 6];
      b7 = X[k * ldx + c0 + 7];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float x = X[k * ldx + lane + i * 32];
        acc[i][0] += x * b0;
        acc[i][1] += x * b1;
        acc[i][2] += x * b2;
        acc[i][3] += x * b3;
        acc[i][4] += x * b4;
        acc[i][5] += x * b5;
        acc[i][6] += x * b6;
        acc[i][7] += x * b7;
      }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float* crow = C + (lane + i * 32) * ldc + c0;
      #pragma unroll
      for (int u = 0; u < W; ++u) crow[u] = acc[i][u];
    }
  }
}

// J = B * Inv, same warp and row mapping as gemm_tn_full. B[row, k] is a
// conflict-free shared read across the warp. A single chain of 128 products
// misses the 1e-6 tolerance on J once beta-scale is 50 or 100, so every 8
// values of k are flushed into a second accumulator. n is a multiple of 8
// and k runs through 127, so the last partial is flushed and nothing is left
// in the panel registers.
__device__ void gemm_row_panel(const float* B, int ldb, const float* Inv, int ldi,
                               float* J, int n, int tid) {
  constexpr int W = 8;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int nwarps = kThreads >> 5;
  const int panels = n / W;
  for (int p = warp; p < panels; p += nwarps) {
    const int c0 = p * W;
    float acc[4][W] = {};
    float part[4][W] = {};
    for (int k = 0; k < n; ++k) {
      float b0, b1, b2, b3, b4, b5, b6, b7;
      b0 = Inv[k * ldi + c0 + 0];
      b1 = Inv[k * ldi + c0 + 1];
      b2 = Inv[k * ldi + c0 + 2];
      b3 = Inv[k * ldi + c0 + 3];
      b4 = Inv[k * ldi + c0 + 4];
      b5 = Inv[k * ldi + c0 + 5];
      b6 = Inv[k * ldi + c0 + 6];
      b7 = Inv[k * ldi + c0 + 7];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float x = B[(lane + i * 32) * ldb + k];
        part[i][0] += x * b0;
        part[i][1] += x * b1;
        part[i][2] += x * b2;
        part[i][3] += x * b3;
        part[i][4] += x * b4;
        part[i][5] += x * b5;
        part[i][6] += x * b6;
        part[i][7] += x * b7;
      }
      if ((k & 7) == 7) {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
          #pragma unroll
          for (int u = 0; u < W; ++u) {
            acc[i][u] += part[i][u];
            part[i][u] = 0.f;
          }
      }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float* out = J + (lane + i * 32) * n + c0;
      #pragma unroll
      for (int u = 0; u < W; ++u) out[u] = acc[i][u];
    }
  }
}

// Stages, all fp32, all inside this block:
//   1. Load the lower triangle of A, add I on the diagonal, zero the upper.
//   2. chol_lower.
//   3. Invert the factor by blocks.
//      inv([A 0; C B]) = [invA 0; -invB * C * invA, invB].
//      Four warps invert the diagonal 32s. Each 64-block then couples its two
//      32s, and one more coupling joins the two 64s. Scratch is 64x64, not a
//      second 128x128, which is what keeps the block under the two-per-SM budget.
//   4. inv = Linv^T Linv, stored in global T. The upper of Linv is still 0.
//   5. Load B over L and form J. Scale T's rows by alpha. J is not scaled.
__global__ void
vdn_delta_factors(const float* __restrict__ A, const float* __restrict__ B,
                  const float* __restrict__ alpha, float* __restrict__ T,
                  float* __restrict__ J, float* __restrict__ scratch, int nmat,
                  int ld, int scratch_local, int* pivot_fail) {
  constexpr int N = kDim;
  const int LD = ld;
  extern __shared__ __align__(16) char smem[];
  float* L = reinterpret_cast<float*>(smem);
  // On-chip when the launch reserved the padded layout. Otherwise one 64x64
  // global scratch per matrix, the same buffer HIP and SYCL use.
  float* Scratch = scratch_local ? (L + static_cast<size_t>(LD) * N)
                                 : (scratch + static_cast<size_t>(blockIdx.x) * 64 * 64);

  const int m = blockIdx.x;
  if (m >= nmat) return;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const float* Am = A + static_cast<size_t>(m) * N * N;
  const float* Bm = B + static_cast<size_t>(m) * N * N;
  float* Tm = T + static_cast<size_t>(m) * N * N;
  float* Jm = J + static_cast<size_t>(m) * N * N;

  const float4* A4 = reinterpret_cast<const float4*>(Am);
  for (int idx = tid; idx < (N * N) / 4; idx += kThreads) {
    const float4 v = A4[idx];
    const int base = idx * 4;
    const int r = base >> 7;
    const int c = base & 127;
    float vals[4] = {v.x, v.y, v.z, v.w};
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int cc = c + u;
      float x = vals[u];
      if (r == cc) x += 1.0f;
      L[r * LD + cc] = (r >= cc) ? x : 0.0f;
    }
  }
  __syncthreads();

  chol_lower(L, LD, N, tid, pivot_fail);
  __syncthreads();

  if (warp < 4) warp_trinv32(L, LD, warp * kNb, lane);
  __syncthreads();
  for (int b = 0; b < 2; ++b) {
    const int r0 = b * 64;
    gemm_nn(L + (r0 + kNb) * LD + r0, LD, L + r0 * LD + r0, LD,
            Scratch, kNb, kNb, kNb, kNb, tid, 1.0f);
    __syncthreads();
    gemm_nn(L + (r0 + kNb) * LD + (r0 + kNb), LD, Scratch, kNb,
            L + (r0 + kNb) * LD + r0, LD, kNb, kNb, kNb, tid, -1.0f);
    __syncthreads();
  }
  gemm_nn(L + 64 * LD, LD, L, LD, Scratch, 64, 64, 64, 64, tid, 1.0f);
  __syncthreads();
  gemm_nn(L + 64 * LD + 64, LD, Scratch, 64, L + 64 * LD, LD, 64, 64, 64, tid, -1.0f);
  __syncthreads();

  gemm_tn_full(L, LD, Tm, N, N, tid);
  __syncthreads();

  const float4* B4 = reinterpret_cast<const float4*>(Bm);
  for (int idx = tid; idx < (N * N) / 4; idx += kThreads) {
    const float4 v = B4[idx];
    const int base = idx * 4;
    const int r = base >> 7;
    const int c = base & 127;
    L[r * LD + c] = v.x;
    L[r * LD + c + 1] = v.y;
    L[r * LD + c + 2] = v.z;
    L[r * LD + c + 3] = v.w;
  }
  __syncthreads();
  gemm_row_panel(L, LD, Tm, N, Jm, N, tid);
  __syncthreads();
  // B has been consumed. Reuse the dead factor buffer for alpha so the row
  // scale is a shared-memory broadcast without a separate allocation.
  if (tid < N) L[tid] = alpha[static_cast<size_t>(m) * N + tid];
  __syncthreads();
  for (int idx = tid; idx < N * N; idx += kThreads) {
    const int r = idx >> 7;
    Tm[idx] *= L[r];
  }
}

namespace {
int g_nmat = 0;
int g_ld = 128;
int g_scratch_local = 0;
size_t g_local_bytes = kVdiTightBytes;
size_t g_device_max = 0;
float* g_scratch = nullptr;
int* g_pivot = nullptr;
char g_desc[192] = "local memory not queried";

size_t device_local_max(const cudaDeviceProp& prop) {
  size_t cap = prop.sharedMemPerBlock;
  if (prop.sharedMemPerBlockOptin > cap) cap = prop.sharedMemPerBlockOptin;
  return cap;
}

int bind_layout() {
  if (g_device_max != 0) return 0;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    fprintf(stderr, "fused: cudaGetDeviceProperties failed\n");
    return 1;
  }
  g_device_max = device_local_max(prop);
  VdiLocalLayout lay{};
  if (!vdi_bind_local(g_device_max, lay)) {
    g_device_max = 0;
    return 1;
  }
  g_ld = lay.ld;
  g_local_bytes = lay.local_bytes;
  g_scratch_local = lay.scratch_local;
  vdi_format_layout(g_desc, sizeof(g_desc), lay, g_device_max);
  return 0;
}
}  // namespace

int fused_wave() { return 32; }

const char* fused_layout() {
  if (bind_layout() != 0) return "local memory query failed";
  return g_desc;
}

int fused_init(int nmat) {
  if (nmat < 1) return 1;
  if (bind_layout() != 0) return 1;
  // Opt in to the bytes this launch will request, and ask the SM to give this
  // kernel shared memory rather than L1. The device ceiling was queried above;
  // a larger ceiling does not add another buffer.
  if (cudaFuncSetAttribute(vdn_delta_factors, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(g_local_bytes)) != cudaSuccess ||
      cudaFuncSetAttribute(vdn_delta_factors, cudaFuncAttributePreferredSharedMemoryCarveout,
                           100) != cudaSuccess) {
    fprintf(stderr, "fused: cudaFuncSetAttribute(%zu) failed\n", g_local_bytes);
    return 1;
  }
  fused_shutdown();
  g_scratch = nullptr;
  if (!g_scratch_local) {
    const size_t bytes = static_cast<size_t>(nmat) * 64 * 64 * sizeof(float);
    if (cudaMalloc(&g_scratch, bytes) != cudaSuccess) {
      fprintf(stderr, "fused: scratch alloc failed\n");
      return 1;
    }
  }
  g_pivot = nullptr;
  if (cudaMalloc(&g_pivot, sizeof(int)) != cudaSuccess ||
      cudaMemset(g_pivot, 0, sizeof(int)) != cudaSuccess) {
    fprintf(stderr, "fused: pivot flag alloc failed\n");
    fused_shutdown();
    return 1;
  }
  g_nmat = nmat;
  return 0;
}

void fused_shutdown() {
  if (g_scratch) cudaFree(g_scratch);
  if (g_pivot) cudaFree(g_pivot);
  g_scratch = nullptr;
  g_pivot = nullptr;
  g_nmat = 0;
}

int fused_pivot_failed() {
  int host = 1;
  if (!g_pivot || cudaMemcpy(&host, g_pivot, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
    return 1;
  return host;
}

int fused_launch(const float* A, const float* B, const float* alpha, float* T, float* J,
                 int nmat) {
  if (nmat != g_nmat || !g_pivot || (!g_scratch_local && !g_scratch)) {
    fprintf(stderr, "fused: launch before init\n");
    return 1;
  }
  vdn_delta_factors<<<nmat, kThreads, g_local_bytes>>>(
      A, B, alpha, T, J, g_scratch, nmat, g_ld, g_scratch_local, g_pivot);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "fused: launch failed\n");
    return 1;
  }
  return 0;
}
