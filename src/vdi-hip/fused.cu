// One block per matrix, one launch. Shared memory holds the lower Cholesky
// factor of I+A and then its inverse, plus a 64x64 scratch for the block
// couplings when the device has room. The inverse is written to global T, B
// is loaded over the factor, and J = B * inv is written to global memory. T
// is then scaled by alpha.
//
// Wave is the hardware wavefront: each lane owns one column of a Wave x Wave
// diagonal tile. Coupling then joins those tiles up to 128 with the block
// formula inv([A 0; C B]) = [invA 0; -invB*C*invA, invB].
//
// Local memory is queried at runtime. local_layout.hpp picks ld 129 with the
// scratch on chip at 82432 bytes, or ld 128 with that scratch in global
// memory at 64KB. Devices below 64KB are rejected. hipFuncSetAttribute opts
// in to whichever of those two sizes the device can hold.

#include "fused.h"
#include "local_layout.hpp"

#include <cstdio>

constexpr int kDim = 128;
constexpr int kThreads = 256;

__device__ void chol_lower(float* L, int ld, int n, int tid, int* pivot_fail) {
  // Same right-looking factorization as the CUDA kernel: two threads per row,
  // each applying a 4-wide update to half of the columns (k, row].
  const int row = tid & (n - 1);
  const int part = tid >> 7;
  float* rp = L + row * ld;
  for (int k = 0; k < n; ++k) {
    if (tid == k) {
      const float d = L[k * ld + k];
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

// Same inverse as CUDA warp_trinv32. Lane `lane` owns column `lane`: it loads
// that column into registers, solves L x = e_lane, and writes the column back.
// The sum for row i only includes columns j with lane <= j < i. L[i, j] is
// broadcast from the lane that owns column j. __shfl reaches every lane of a
// 32- or 64-wide wavefront, so both widths share this path. No barrier inside
// the tile: every read of the factor finished before any column is stored.
template <int Wave>
__device__ void wave_trinv(float* S, int ld, int r0, int lane) {
  float colv[Wave];
  #pragma unroll 1
  for (int i = 0; i < Wave; ++i)
    colv[i] = S[(r0 + i) * ld + (r0 + lane)];

  float x[Wave];
  #pragma unroll 1
  for (int i = 0; i < Wave; ++i) x[i] = 0.0f;
  x[lane] = 1.0f / colv[lane];

  #pragma unroll 1
  for (int i = 1; i < Wave; ++i) {
    float sum = 0.0f;
    #pragma unroll 1
    for (int j = 0; j < i; ++j) {
      const float Lij = __shfl(colv[i], j, Wave);
      if (j >= lane) sum += Lij * x[j];
    }
    const float Lii = __shfl(colv[i], i, Wave);
    if (i > lane) x[i] = -sum / Lii;
  }

  #pragma unroll 1
  for (int i = lane; i < Wave; ++i)
    S[(r0 + i) * ld + (r0 + lane)] = x[i];
}

__device__ void gemm_nn(const float* A, int lda, const float* B, int ldb,
                        float* C, int ldc, int M, int N, int K, int tid,
                        float scale) {
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

// C = X^T X. A lane owns rows lane + i*Wave, which covers 0..127 whether
// Wave is 32 (four rows) or 64 (two rows).
template <int Wave>
__device__ void gemm_tn_full(const float* X, int ldx, float* C, int ldc,
                             int n, int tid) {
  constexpr int W = 8;
  constexpr int kRows = kDim / Wave;
  const int wave = tid / Wave;
  const int lane = tid - wave * Wave;
  const int nwaves = kThreads / Wave;
  const int panels = n / W;
  for (int p = wave; p < panels; p += nwaves) {
    const int c0 = p * W;
    float acc[kRows][W];
    #pragma unroll
    for (int i = 0; i < kRows; ++i)
      #pragma unroll
      for (int u = 0; u < W; ++u) acc[i][u] = 0.f;
    for (int k = 0; k < n; ++k) {
      float b[W];
      #pragma unroll
      for (int u = 0; u < W; ++u) b[u] = X[k * ldx + c0 + u];
      #pragma unroll
      for (int i = 0; i < kRows; ++i) {
        const float x = X[k * ldx + lane + i * Wave];
        #pragma unroll
        for (int u = 0; u < W; ++u) acc[i][u] += x * b[u];
      }
    }
    #pragma unroll
    for (int i = 0; i < kRows; ++i) {
      float* crow = C + (lane + i * Wave) * ldc + c0;
      #pragma unroll
      for (int u = 0; u < W; ++u) crow[u] = acc[i][u];
    }
  }
}

// J = B * Inv. Partial sums of 8 keep the fp32 dot inside 1e-6 at beta-scale
// 50 and 100. n is a multiple of 8, so k == 127 flushes the last partial.
template <int Wave>
__device__ void gemm_row_panel(const float* B, int ldb, const float* Inv, int ldi,
                               float* J, int n, int tid) {
  constexpr int W = 8;
  constexpr int kRows = kDim / Wave;
  const int wave = tid / Wave;
  const int lane = tid - wave * Wave;
  const int nwaves = kThreads / Wave;
  const int panels = n / W;
  for (int p = wave; p < panels; p += nwaves) {
    const int c0 = p * W;
    float acc[kRows][W];
    float part[kRows][W];
    #pragma unroll
    for (int i = 0; i < kRows; ++i)
      #pragma unroll
      for (int u = 0; u < W; ++u) acc[i][u] = part[i][u] = 0.f;
    for (int k = 0; k < n; ++k) {
      float b[W];
      #pragma unroll
      for (int u = 0; u < W; ++u) b[u] = Inv[k * ldi + c0 + u];
      #pragma unroll
      for (int i = 0; i < kRows; ++i) {
        const float x = B[(lane + i * Wave) * ldb + k];
        #pragma unroll
        for (int u = 0; u < W; ++u) part[i][u] += x * b[u];
      }
      if ((k & 7) == 7) {
        #pragma unroll
        for (int i = 0; i < kRows; ++i)
          #pragma unroll
          for (int u = 0; u < W; ++u) {
            acc[i][u] += part[i][u];
            part[i][u] = 0.f;
          }
      }
    }
    #pragma unroll
    for (int i = 0; i < kRows; ++i) {
      float* out = J + (lane + i * Wave) * n + c0;
      #pragma unroll
      for (int u = 0; u < W; ++u) out[u] = acc[i][u];
    }
  }
}

template <int Wave>
__global__ void
vdn_delta_factors(const float* __restrict__ A, const float* __restrict__ B,
                  const float* __restrict__ alpha, float* __restrict__ T,
                  float* __restrict__ J, float* __restrict__ scratch, int nmat,
                  int ld, int scratch_local, int* pivot_fail) {
  constexpr int N = kDim;
  const int LD = ld;
  extern __shared__ __align__(16) char smem[];
  float* L = reinterpret_cast<float*>(smem);
  // On-chip when the device opted in to at least 82432 bytes. Otherwise one
  // 64x64 global scratch per matrix; couplings of 32 reuse it after a barrier.
  float* Scratch = scratch_local ? (L + static_cast<size_t>(LD) * N)
                                 : (scratch + static_cast<size_t>(blockIdx.x) * 64 * 64);

  const int m = blockIdx.x;
  if (m >= nmat) return;
  const int tid = threadIdx.x;
  const int lane = tid % Wave;
  const int wave = tid / Wave;

  const float* Am = A + static_cast<size_t>(m) * N * N;
  const float* Bm = B + static_cast<size_t>(m) * N * N;
  float* Tm = T + static_cast<size_t>(m) * N * N;
  float* Jm = J + static_cast<size_t>(m) * N * N;
  const float* arow = alpha + static_cast<size_t>(m) * N;

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

  // One wave per diagonal tile. A wave of 32 covers four tiles with the
  // first four waves; a wave of 64 covers two tiles with the first two.
  if (wave < N / Wave)
    wave_trinv<Wave>(L, LD, wave * Wave, lane);
  __syncthreads();

  for (int blk = Wave; blk < N; blk *= 2) {
    const int npairs = N / (2 * blk);
    for (int p = 0; p < npairs; ++p) {
      const int r0 = p * 2 * blk;
      gemm_nn(L + (r0 + blk) * LD + r0, LD, L + r0 * LD + r0, LD, Scratch, blk,
              blk, blk, blk, tid, 1.0f);
      __syncthreads();
      gemm_nn(L + (r0 + blk) * LD + (r0 + blk), LD, Scratch, blk,
              L + (r0 + blk) * LD + r0, LD, blk, blk, blk, tid, -1.0f);
      __syncthreads();
    }
  }

  gemm_tn_full<Wave>(L, LD, Tm, N, N, tid);
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
  gemm_row_panel<Wave>(L, LD, Tm, N, Jm, N, tid);
  __syncthreads();
  // B has been consumed. Reuse the dead factor buffer for alpha so the row
  // scale is a shared-memory broadcast without a separate allocation.
  if (tid < N) L[tid] = arow[tid];
  __syncthreads();
  for (int idx = tid; idx < N * N; idx += kThreads) {
    const int r = idx >> 7;
    Tm[idx] *= L[r];
  }
}

namespace {
int g_wave = 0;
int g_nmat = 0;
int g_ld = 128;
int g_scratch_local = 0;
size_t g_local_bytes = kVdiTightBytes;
size_t g_device_max = 0;
float* g_scratch = nullptr;
int* g_pivot = nullptr;
char g_desc[192] = "local memory not queried";

size_t device_local_max(const hipDeviceProp_t& prop) {
  size_t cap = prop.sharedMemPerBlock;
  if (prop.sharedMemPerBlockOptin > cap) cap = prop.sharedMemPerBlockOptin;
  return cap;
}

int bind_layout() {
  if (g_device_max != 0) return 0;
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    fprintf(stderr, "fused: hipGetDeviceProperties failed\n");
    return 1;
  }
  g_wave = prop.warpSize;
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

int fused_wave() { return g_wave; }

const char* fused_layout() {
  if (bind_layout() != 0) return "local memory query failed";
  return g_desc;
}

int fused_init(int nmat) {
  if (nmat < 1) return 1;
  if (bind_layout() != 0) return 1;
  if (g_wave != 32 && g_wave != 64) {
    fprintf(stderr, "fused: wave size %d is not 32 or 64\n", g_wave);
    return 1;
  }
  auto* fn = (g_wave == 64) ? vdn_delta_factors<64> : vdn_delta_factors<32>;
  // Opt in up to the bytes this launch will request. The device ceiling was
  // queried above; setting a larger attribute would not add a second buffer.
  hipError_t attr = hipFuncSetAttribute(
      reinterpret_cast<const void*>(fn), hipFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(g_local_bytes));
  if (attr != hipSuccess) {
    fprintf(stderr, "fused: hipFuncSetAttribute(%zu) failed\n", g_local_bytes);
    return 1;
  }
  fused_shutdown();
  g_scratch = nullptr;
  if (!g_scratch_local) {
    const size_t bytes = static_cast<size_t>(nmat) * 64 * 64 * sizeof(float);
    if (hipMalloc(&g_scratch, bytes) != hipSuccess) {
      fprintf(stderr, "fused: scratch alloc failed\n");
      return 1;
    }
  }
  g_pivot = nullptr;
  if (hipMalloc(&g_pivot, sizeof(int)) != hipSuccess ||
      hipMemset(g_pivot, 0, sizeof(int)) != hipSuccess) {
    fprintf(stderr, "fused: pivot flag alloc failed\n");
    fused_shutdown();
    return 1;
  }
  g_nmat = nmat;
  return 0;
}

void fused_shutdown() {
  if (g_scratch) (void)hipFree(g_scratch);
  if (g_pivot) (void)hipFree(g_pivot);
  g_scratch = nullptr;
  g_pivot = nullptr;
  g_nmat = 0;
}

int fused_pivot_failed() {
  int host = 1;
  if (!g_pivot || hipMemcpy(&host, g_pivot, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  return host;
}

int fused_launch(const float* A, const float* B, const float* alpha,
                 float* T, float* J, int nmat) {
  if (nmat != g_nmat || !g_pivot || (!g_scratch_local && !g_scratch)) {
    fprintf(stderr, "fused: launch before init\n");
    return 1;
  }
  if (g_wave == 64) {
    vdn_delta_factors<64><<<nmat, kThreads, g_local_bytes>>>(
        A, B, alpha, T, J, g_scratch, nmat, g_ld, g_scratch_local, g_pivot);
  } else {
    vdn_delta_factors<32><<<nmat, kThreads, g_local_bytes>>>(
        A, B, alpha, T, J, g_scratch, nmat, g_ld, g_scratch_local, g_pivot);
  }
  if (hipGetLastError() != hipSuccess) {
    fprintf(stderr, "fused: launch failed\n");
    return 1;
  }
  return 0;
}
