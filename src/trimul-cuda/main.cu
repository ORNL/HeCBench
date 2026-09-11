/*
 * Triangle Multiplicative Update (TriMul) forward pass -- BF16.
 *
 * A key component of the AlphaFold2/3 Evoformer, benchmarked here after
 * NVIDIA cuEquivariance's triangle_multiplicative_update (Apache-2.0).
 * See reference.h for the algorithm and tensor layouts.
 *
 * The sigmoid-gated dual GEMMs are single WMMA (tensor core) kernels whose
 * epilogue does bias + sigmoid gate + mask + BF16 cast + scatter while the
 * accumulators are still in registers, and LayerNorm emits BF16 directly, so
 * no projection/gate FP32 scratch is ever written to HBM. The triangle
 * projection is a cuBLAS batched BF16 GEMM.
 *
 * LayerNorm and epilogue arithmetic use FP32, GEMM operands are BF16, and all
 * matrix products accumulate into FP32.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include "reference.h"

using namespace nvcuda;

#define CUDA_CHECK(x) cuda_check((x), #x, __FILE__, __LINE__)
static void cuda_check(cudaError_t e, const char* call, const char* file, int line) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s: %s\n",
            file, line, call, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

#define CUBLAS_CHECK(x) cublas_check((x), #x, __FILE__, __LINE__)
static void cublas_check(cublasStatus_t e, const char* call,
                         const char* file, int line) {
  if (e != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS error at %s:%d: %s: status %d\n",
            file, line, call, (int)e);
    exit(EXIT_FAILURE);
  }
}

static float* make_random_float(size_t n) {
  float* a = (float*)xmalloc(n * sizeof(float));
  for (size_t i = 0; i < n; i++)
    a[i] = rand() / (float)RAND_MAX * 2.f - 1.f;
  return a;
}

static __nv_bfloat16* make_bfloat16(const float* src, size_t n) {
  __nv_bfloat16* dst = (__nv_bfloat16*)xmalloc(n * sizeof(__nv_bfloat16));
  for (size_t i = 0; i < n; i++) dst[i] = __float2bfloat16(src[i]);
  return dst;
}

// The BF16 operands only keep 8 significant bits, so the result is compared
// against the FP32 reference through error statistics (max abs, max rel, RMS)
// rather than an FP32-tight per-element tolerance.
static bool verify_result(const float* result, const float* ref,
                          const char* name, size_t n) {
  float* h = (float*)xmalloc(n * sizeof(float));
  CUDA_CHECK(cudaMemcpy(h, result, n * sizeof(float), cudaMemcpyDeviceToHost));
  double max_abs = 0, max_rel = 0, max_scaled = 0, sse = 0, sref = 0;
  for (size_t i = 0; i < n; ++i) {
    const double e = fabs((double)ref[i] - h[i]);
    max_abs = fmax(max_abs, e);
    max_rel = fmax(max_rel, e / (fabs((double)ref[i]) + 1e-6));
    max_scaled = fmax(max_scaled, e / (1.0 + fabs((double)ref[i])));
    sse += e * e;
    sref += (double)ref[i] * ref[i];
  }
  free(h);
  const double rms_rel = sqrt(sse / (sref + 1e-12));
  printf("%s: max_abs=%.4g  max_rel=%.4g  max_scaled=%.4g  "
         "rms_rel=%.4g (tolerance 5e-2)\n",
         name, max_abs, max_rel, max_scaled, rms_rel);
  return max_scaled < 5e-2 && rms_rel < 5e-2;
}

// expf(-v) overflows for large negative v, so branch on the sign.
__device__ __forceinline__ float sigmoidf_dev(float v) {
  if (v >= 0.f) return 1.0f / (1.0f + expf(-v));
  const float e = expf(v);
  return e / (1.0f + e);
}

// Both LayerNorms give a row to a warp rather than to a whole block, so the
// reduction stays inside the warp: no shared memory, no __syncthreads, and no
// idle lanes when D is smaller than a block.  Sum and sum-of-squares are
// accumulated in the same pass and reduced together, which reads the row twice
// instead of three times, and the xor butterfly leaves the totals in every
// lane so no broadcast is needed.
template <int W>
__device__ __forceinline__ void warp_sum2(float& s, float& ss) {
  for (int off = W / 2; off > 0; off >>= 1) {
    s += __shfl_xor_sync(0xffffffffu, s, off);
    ss += __shfl_xor_sync(0xffffffffu, ss, off);
  }
}

// LayerNorm over D for contiguous rows, emitting BF16 directly so the
// normalized activations never round-trip through FP32 memory.
template <int TPB, int W>
__global__ void layernorm_contig_bf16_kernel(const float* __restrict__ in,
                                             __nv_bfloat16* __restrict__ out,
                                             const float* __restrict__ w,
                                             const float* __restrict__ b,
                                             int rows, int D, float eps) {
  const int lane = threadIdx.x % W;
  const int row = blockIdx.x * (TPB / W) + threadIdx.x / W;
  if (row >= rows) return;
  const float* x = in + (size_t)row * D;
  __nv_bfloat16* y = out + (size_t)row * D;

  float s = 0.f, ss = 0.f;
  for (int d = lane; d < D; d += W) {
    const float v = x[d];
    s += v;
    ss += v * v;
  }
  warp_sum2<W>(s, ss);

  const float mean = s / D;
  const float rstd = rsqrtf(ss / D - mean * mean + eps);

  for (int d = lane; d < D; d += W)
    y[d] = __float2bfloat16((x[d] - mean) * rstd * w[d] + b[d]);
}

// Gathering LayerNorm, emitting BF16 directly.
template <int TPB, int W>
__global__ void layernorm_gather_bf16_kernel(const float* __restrict__ tri,
                                             __nv_bfloat16* __restrict__ xout,
                                             const float* __restrict__ w,
                                             const float* __restrict__ b,
                                             int rows, int N, int D, float eps) {
  const int lane = threadIdx.x % W;
  const int row = blockIdx.x * (TPB / W) + threadIdx.x / W;
  if (row >= rows) return;
  const int NN = N * N;
  const size_t base = (size_t)(row / NN) * D * NN + (row % NN);

  float s = 0.f, ss = 0.f;
  for (int d = lane; d < D; d += W) {
    const float v = tri[base + (size_t)d * NN];
    s += v;
    ss += v * v;
  }
  warp_sum2<W>(s, ss);

  const float mean = s / D;
  const float rstd = rsqrtf(ss / D - mean * mean + eps);

  __nv_bfloat16* y = xout + (size_t)row * D;
  for (int d = lane; d < D; d += W)
    y[d] = __float2bfloat16((tri[base + (size_t)d * NN] - mean) * rstd * w[d] + b[d]);
}

// --- WMMA tiling for the fused gated dual GEMMs -----------------------------
// Each warp owns an (FM*16) x (FN*16) output quadrant and carries *two*
// accumulator sets (proj + gate) that share the same A fragments, so the
// sigmoid gate needs no cross-thread traffic.
namespace fused {

template <int BM_, int BN_, int BK_, int WARPS_M_, int WARPS_N_>
struct Cfg {
  static constexpr int BM = BM_, BN = BN_, BK = BK_;
  static constexpr int WARPS_M = WARPS_M_, WARPS_N = WARPS_N_;
  static constexpr int NWARPS = WARPS_M * WARPS_N;
  static constexpr int NTHREADS = NWARPS * 32;
  static constexpr int FM = BM / (WARPS_M * 16);
  static constexpr int FN = BN / (WARPS_N * 16);
  // WMMA ldm must be a multiple of 16 bytes: 8 elements bf16, 4 elements fp32.
  static constexpr int LDA = BK + 8;   // bf16 staging row stride
  static constexpr int LDT = 16 + 4;   // fp32 per-warp epilogue tile stride

  static constexpr size_t stage_in  = (size_t)(BM * LDA + 2 * BN * LDA) * sizeof(__nv_bfloat16);
  static constexpr size_t stage_out = (size_t)(2 * BM * LDA + 2 * BN * LDA) * sizeof(__nv_bfloat16);
  // The epilogue reuses the (now dead) staging memory.  Only one 16x16 tile
  // pair per warp is live at a time, limiting the shared-memory footprint.
  static constexpr size_t epi = (size_t)NWARPS * 2 * 16 * LDT * sizeof(float);

  // Staging buffers per k-step: the cp.async pipeline keeps the next tile in
  // flight while the current one feeds the MMAs, so it needs two.  The HIP
  // port stages synchronously and sets this to one; everything else about the
  // shared-memory layout is the same in both.
  static constexpr int STAGES = 2;
  static constexpr size_t smem_in  = STAGES * stage_in  > epi ? STAGES * stage_in  : epi;
  static constexpr size_t smem_out = STAGES * stage_out > epi ? STAGES * stage_out : epi;

  // Staying within the 48 KB that a block gets without an explicit opt-in
  // keeps the kernels launchable on every BF16-capable device.
  static_assert(smem_in <= 48 * 1024, "input tile exceeds the default shared memory limit");
  static_assert(smem_out <= 48 * 1024, "output tile exceeds the default shared memory limit");

  static_assert(BM % (WARPS_M * 16) == 0, "BM must tile into WARPS_M*16");
  static_assert(BN % (WARPS_N * 16) == 0, "BN must tile into WARPS_N*16");
  static_assert(BK % 16 == 0, "BK must be a multiple of the WMMA k-step");
  static_assert(BK % 8 == 0, "BK must be a multiple of the 16-byte vector width");
};

// Stage a ROWS x BK tile of a row-major (*, D) BF16 matrix into shared memory,
// zero-padding out-of-range elements.  Uses 16-byte vector copies when the
// alignment allows it, else falls back to scalar.
template <int ROWS, int BK, int LDA, int NTHREADS>
__device__ __forceinline__ void stage_sync(const __nv_bfloat16* __restrict__ src,
                                           int ld_src, int row0, int row_limit,
                                           int kt, int D, __nv_bfloat16* dst, int tid) {
  constexpr int VW = 8;              // bf16 per 16-byte vector
  constexpr int VPR = BK / VW;
  // The alignment of every vector load depends on the source pitch, not on
  // the k-limit; a partial row tile still takes this path and zero-fills the
  // out-of-range rows below.
  if ((ld_src % VW) == 0 && kt + BK <= D) {
    for (int idx = tid; idx < ROWS * VPR; idx += NTHREADS) {
      const int r = idx / VPR, v = idx % VPR;
      const int gr = row0 + r;
      uint4 val = make_uint4(0u, 0u, 0u, 0u);
      if (gr < row_limit)
        val = *reinterpret_cast<const uint4*>(src + (size_t)gr * ld_src + kt + v * VW);
      *reinterpret_cast<uint4*>(dst + r * LDA + v * VW) = val;
    }
  } else {
    const __nv_bfloat16 zero = __float2bfloat16(0.f);
    for (int idx = tid; idx < ROWS * BK; idx += NTHREADS) {
      const int r = idx / BK, k = idx % BK;
      const int gr = row0 + r, gk = kt + k;
      dst[r * LDA + k] = (gr < row_limit && gk < D) ? src[(size_t)gr * ld_src + gk] : zero;
    }
  }
}

// Same tile, issued as cp.async 16-byte copies.  Caller must guarantee the
// whole tile is in range and 16-byte aligned (see the `full` predicate).
template <int ROWS, int BK, int LDA, int NTHREADS>
__device__ __forceinline__ void stage_async(const __nv_bfloat16* __restrict__ src,
                                            int ld_src, int row0, int kt,
                                            __nv_bfloat16* dst, int tid) {
  constexpr int VW = 8;
  constexpr int VPR = BK / VW;
  for (int idx = tid; idx < ROWS * VPR; idx += NTHREADS) {
    const int r = idx / VPR, v = idx % VPR;
    __pipeline_memcpy_async(dst + r * LDA + v * VW,
                            src + (size_t)(row0 + r) * ld_src + kt + v * VW, 16);
  }
}

}  // namespace fused

// Fused sigmoid-gated dual GEMM (input side), tensor cores, BF16 in / BF16 out.
//
//   proj[m,c] = xn[m,:] . p_in[c,:] + pb[c]
//   gate[m,c] = xn[m,:] . g_in[c,:] + gb[c]
//   v         = sigmoid(gate) * proj * mask[m]
//   c <  D -> a_out[(b*D + c    )*NN + ik]
//   c >= D -> b_out[(b*D + c - D)*NN + ik]     with m = b*NN + ik
//
// Both accumulator sets reuse one A fragment, and the whole element-wise tail
// runs on registers/shared -- proj and gate never touch HBM.
template <typename C>
__global__ __launch_bounds__(C::NTHREADS)
void gated_gemm_in_fused_kernel(const __nv_bfloat16* __restrict__ A,
                                const __nv_bfloat16* __restrict__ Pw,
                                const __nv_bfloat16* __restrict__ Gw,
                                const float* __restrict__ pb,
                                const float* __restrict__ gb,
                                const float* __restrict__ mask,
                                __nv_bfloat16* __restrict__ a_out,
                                __nv_bfloat16* __restrict__ b_out,
                                int rows, int D, int NN) {
  using fused::stage_sync;
  using fused::stage_async;
  constexpr int BM = C::BM, BN = C::BN, BK = C::BK, LDA = C::LDA, LDT = C::LDT;
  constexpr int FM = C::FM, FN = C::FN, NT = C::NTHREADS;
  constexpr int stage_elems = BM * LDA + 2 * BN * LDA;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* sh = reinterpret_cast<__nv_bfloat16*>(smem_raw);

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int warp_m = warp / C::WARPS_N, warp_n = warp % C::WARPS_N;
  const int m0 = blockIdx.y * BM;
  const int c0 = blockIdx.x * BN;
  const int twoD = 2 * D;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accP[FM][FN], accG[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) {
      wmma::fill_fragment(accP[i][j], 0.f);
      wmma::fill_fragment(accG[i][j], 0.f);
    }

  // One MMA sweep over a staged A/projection-weight/gate-weight tile triple.
  auto mma_stage = [&](__nv_bfloat16* As, __nv_bfloat16* Ps, __nv_bfloat16* Gs) {
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> pf[FN], gf[FN];
#pragma unroll
      for (int i = 0; i < FM; i++)
        wmma::load_matrix_sync(af[i], &As[(warp_m * (FM * 16) + i * 16) * LDA + kk], LDA);
#pragma unroll
      for (int j = 0; j < FN; j++) {
        wmma::load_matrix_sync(pf[j], &Ps[(warp_n * (FN * 16) + j * 16) * LDA + kk], LDA);
        wmma::load_matrix_sync(gf[j], &Gs[(warp_n * (FN * 16) + j * 16) * LDA + kk], LDA);
      }
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) {
          wmma::mma_sync(accP[i][j], af[i], pf[j], accP[i][j]);
          wmma::mma_sync(accG[i][j], af[i], gf[j], accG[i][j]);
        }
    }
  };

  // cp.async needs every access in range and 16-byte aligned; edge blocks and
  // awkward D fall back to the synchronous staging path.
  const bool full = (m0 + BM <= rows) && (c0 + BN <= twoD) &&
                    (D % 8 == 0) && (D % BK == 0);

  if (full) {
    auto issue = [&](int buf, int kt) {
      __nv_bfloat16* s = sh + buf * stage_elems;
      stage_async<BM, BK, LDA, NT>(A, D, m0, kt, s, tid);
      stage_async<BN, BK, LDA, NT>(Pw, D, c0, kt, s + BM * LDA, tid);
      stage_async<BN, BK, LDA, NT>(Gw, D, c0, kt, s + BM * LDA + BN * LDA, tid);
    };
    issue(0, 0);
    __pipeline_commit();
    int buf = 0;
    for (int kt = 0; kt < D; kt += BK, buf ^= 1) {
      const int knext = kt + BK;
      const bool more = knext < D;
      if (more) { issue(buf ^ 1, knext); __pipeline_commit(); }
      __pipeline_wait_prior(more ? 1 : 0);
      __syncthreads();
      __nv_bfloat16* s = sh + buf * stage_elems;
      mma_stage(s, s + BM * LDA, s + BM * LDA + BN * LDA);
      __syncthreads();
    }
  } else {
    for (int kt = 0; kt < D; kt += BK) {
      stage_sync<BM, BK, LDA, NT>(A, D, m0, rows, kt, D, sh, tid);
      stage_sync<BN, BK, LDA, NT>(Pw, D, c0, twoD, kt, D, sh + BM * LDA, tid);
      stage_sync<BN, BK, LDA, NT>(Gw, D, c0, twoD, kt, D, sh + BM * LDA + BN * LDA, tid);
      __syncthreads();
      mma_stage(sh, sh + BM * LDA, sh + BM * LDA + BN * LDA);
      __syncthreads();
    }
  }

  // Epilogue: one 16x16 tile pair at a time in a per-warp shared scratch, so
  // proj/gate are combined, gated, masked, cast and scattered entirely on chip.
  // The staging memory is dead here (all warps passed the final __syncthreads).
  float* Tp = reinterpret_cast<float*>(smem_raw) + warp * (2 * 16 * LDT);
  float* Tg = Tp + 16 * LDT;
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) {
      wmma::store_matrix_sync(Tp, accP[i][j], LDT, wmma::mem_row_major);
      wmma::store_matrix_sync(Tg, accG[i][j], LDT, wmma::mem_row_major);
      __syncwarp();
      const int mbase = m0 + warp_m * (FM * 16) + i * 16;
      const int cbase = c0 + warp_n * (FN * 16) + j * 16;
      // m fast-varying: consecutive lanes write consecutive ik, which is
      // contiguous in the (B,D,N,N) destination.
      for (int t = lane; t < 256; t += 32) {
        const int ml = t & 15, cl = t >> 4;
        const int gm = mbase + ml, gc = cbase + cl;
        if (gm >= rows || gc >= twoD) continue;
        const float proj = Tp[ml * LDT + cl] + pb[gc];
        const float gate = Tg[ml * LDT + cl] + gb[gc];
        const float v = sigmoidf_dev(gate) * proj * mask[gm];
        const int bidx = gm / NN, ik = gm % NN;
        const int d = gc < D ? gc : gc - D;
        __nv_bfloat16* dst = gc < D ? a_out : b_out;
        dst[(size_t)(bidx * D + d) * NN + ik] = __float2bfloat16(v);
      }
      __syncwarp();
    }
}

// Fused sigmoid-gated dual GEMM (output side), tensor cores, BF16 in / FP32 out.
//
//   out[m,dp] = sigmoid(xn[m,:] . g_out[dp,:] + gb) * (xout[m,:] . p_out[dp,:] + pb)
//
// Unlike the input side the two GEMMs have *different* A operands (xn vs xout),
// so two A tiles are staged; the gate/proj combine still stays on-chip.
template <typename C>
__global__ __launch_bounds__(C::NTHREADS)
void gated_gemm_out_fused_kernel(const __nv_bfloat16* __restrict__ Xi,
                                 const __nv_bfloat16* __restrict__ Xo,
                                 const __nv_bfloat16* __restrict__ Pw,
                                 const __nv_bfloat16* __restrict__ Gw,
                                 const float* __restrict__ pb,
                                 const float* __restrict__ gb,
                                 float* __restrict__ out,
                                 int rows, int D) {
  using fused::stage_sync;
  using fused::stage_async;
  constexpr int BM = C::BM, BN = C::BN, BK = C::BK, LDA = C::LDA, LDT = C::LDT;
  constexpr int FM = C::FM, FN = C::FN, NT = C::NTHREADS;
  constexpr int oXo = BM * LDA, oPs = 2 * BM * LDA, oGs = 2 * BM * LDA + BN * LDA;
  constexpr int stage_elems = 2 * BM * LDA + 2 * BN * LDA;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* sh = reinterpret_cast<__nv_bfloat16*>(smem_raw);

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int warp_m = warp / C::WARPS_N, warp_n = warp % C::WARPS_N;
  const int m0 = blockIdx.y * BM;
  const int c0 = blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accP[FM][FN], accG[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) {
      wmma::fill_fragment(accP[i][j], 0.f);
      wmma::fill_fragment(accG[i][j], 0.f);
    }

  auto mma_stage = [&](__nv_bfloat16* s) {
    __nv_bfloat16* Xis = s;
    __nv_bfloat16* Xos = s + oXo;
    __nv_bfloat16* Ps  = s + oPs;
    __nv_bfloat16* Gs  = s + oGs;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> xif[FM], xof[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> pf[FN], gf[FN];
#pragma unroll
      for (int i = 0; i < FM; i++) {
        wmma::load_matrix_sync(xif[i], &Xis[(warp_m * (FM * 16) + i * 16) * LDA + kk], LDA);
        wmma::load_matrix_sync(xof[i], &Xos[(warp_m * (FM * 16) + i * 16) * LDA + kk], LDA);
      }
#pragma unroll
      for (int j = 0; j < FN; j++) {
        wmma::load_matrix_sync(pf[j], &Ps[(warp_n * (FN * 16) + j * 16) * LDA + kk], LDA);
        wmma::load_matrix_sync(gf[j], &Gs[(warp_n * (FN * 16) + j * 16) * LDA + kk], LDA);
      }
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) {
          wmma::mma_sync(accP[i][j], xof[i], pf[j], accP[i][j]);
          wmma::mma_sync(accG[i][j], xif[i], gf[j], accG[i][j]);
        }
    }
  };

  const bool full = (m0 + BM <= rows) && (c0 + BN <= D) &&
                    (D % 8 == 0) && (D % BK == 0);

  if (full) {
    auto issue = [&](int buf, int kt) {
      __nv_bfloat16* s = sh + buf * stage_elems;
      stage_async<BM, BK, LDA, NT>(Xi, D, m0, kt, s, tid);
      stage_async<BM, BK, LDA, NT>(Xo, D, m0, kt, s + oXo, tid);
      stage_async<BN, BK, LDA, NT>(Pw, D, c0, kt, s + oPs, tid);
      stage_async<BN, BK, LDA, NT>(Gw, D, c0, kt, s + oGs, tid);
    };
    issue(0, 0);
    __pipeline_commit();
    int buf = 0;
    for (int kt = 0; kt < D; kt += BK, buf ^= 1) {
      const int knext = kt + BK;
      const bool more = knext < D;
      if (more) { issue(buf ^ 1, knext); __pipeline_commit(); }
      __pipeline_wait_prior(more ? 1 : 0);
      __syncthreads();
      mma_stage(sh + buf * stage_elems);
      __syncthreads();
    }
  } else {
    for (int kt = 0; kt < D; kt += BK) {
      stage_sync<BM, BK, LDA, NT>(Xi, D, m0, rows, kt, D, sh, tid);
      stage_sync<BM, BK, LDA, NT>(Xo, D, m0, rows, kt, D, sh + oXo, tid);
      stage_sync<BN, BK, LDA, NT>(Pw, D, c0, D, kt, D, sh + oPs, tid);
      stage_sync<BN, BK, LDA, NT>(Gw, D, c0, D, kt, D, sh + oGs, tid);
      __syncthreads();
      mma_stage(sh);
      __syncthreads();
    }
  }

  float* Tp = reinterpret_cast<float*>(smem_raw) + warp * (2 * 16 * LDT);
  float* Tg = Tp + 16 * LDT;
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) {
      wmma::store_matrix_sync(Tp, accP[i][j], LDT, wmma::mem_row_major);
      wmma::store_matrix_sync(Tg, accG[i][j], LDT, wmma::mem_row_major);
      __syncwarp();
      const int mbase = m0 + warp_m * (FM * 16) + i * 16;
      const int cbase = c0 + warp_n * (FN * 16) + j * 16;
      // dp fast-varying: out is (rows, D) row-major, so lanes coalesce.
      for (int t = lane; t < 256; t += 32) {
        const int ml = t >> 4, cl = t & 15;
        const int gm = mbase + ml, gdp = cbase + cl;
        if (gm >= rows || gdp >= D) continue;
        const float proj = Tp[ml * LDT + cl] + pb[gdp];
        const float gate = Tg[ml * LDT + cl] + gb[gdp];
        out[(size_t)gm * D + gdp] = sigmoidf_dev(gate) * proj;
      }
      __syncwarp();
    }
}

// ===========================================================================

struct TriMulBuffers {
  float *x, *mask, *tri, *out;
  __nv_bfloat16 *xn, *xout, *a, *b;
  __nv_bfloat16 *p_in_w, *g_in_w, *p_out_w, *g_out_w;
  float *norm_in_w, *norm_in_b, *p_in_b, *g_in_b;
  float *norm_out_w, *norm_out_b, *p_out_b, *g_out_b;
  cublasHandle_t blas;
};

// Triangle projection (BF16 batched GEMM, tensor cores) row-major mapping:
//   outgoing: C[i,j] = sum_k a[i,k] b[j,k]  => Ccm = op(b,T) . op(a,N)
//   incoming: C[i,j] = sum_k a[k,i] b[k,j]  => Ccm = op(b,N) . op(a,T)
template <int OUTGOING>
static void triangle_batched_bf16(const TriMulBuffers& d, int B, int N, int D) {
  const size_t planes = (size_t)B * D, plane_sz = (size_t)N * N;
  const float alpha = 1.f, beta = 0.f;
  const cublasOperation_t ta = OUTGOING ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t tb = OUTGOING ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      d.blas, ta, tb, N, N, N, &alpha,
      d.b, CUDA_R_16BF, N, plane_sz,
      d.a, CUDA_R_16BF, N, plane_sz,
      &beta, d.tri, CUDA_R_32F, N, plane_sz,
      (int)planes, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// One TriMul forward pass: LayerNorm -> gated dual GEMM -> triangle
// projection -> LayerNorm -> gated dual GEMM.
template <typename C, int OUTGOING>
static void trimul_forward(const TriMulBuffers& d, int B, int N, int D, float eps) {
  constexpr int TPB = 256;
  constexpr int W = 32;                       // one warp per normalized row
  const int rows = B * N * N;
  const int NN = N * N;
  const int ln_blocks = (rows + TPB / W - 1) / (TPB / W);

  layernorm_contig_bf16_kernel<TPB, W><<<ln_blocks, TPB>>>(
      d.x, d.xn, d.norm_in_w, d.norm_in_b, rows, D, eps);

  dim3 gi((2 * D + C::BN - 1) / C::BN, (rows + C::BM - 1) / C::BM);
  gated_gemm_in_fused_kernel<C><<<gi, C::NTHREADS, C::smem_in>>>(
      d.xn, d.p_in_w, d.g_in_w, d.p_in_b, d.g_in_b, d.mask,
      d.a, d.b, rows, D, NN);

  triangle_batched_bf16<OUTGOING>(d, B, N, D);

  layernorm_gather_bf16_kernel<TPB, W><<<ln_blocks, TPB>>>(
      d.tri, d.xout, d.norm_out_w, d.norm_out_b, rows, N, D, eps);

  dim3 go((D + C::BN - 1) / C::BN, (rows + C::BM - 1) / C::BM);
  gated_gemm_out_fused_kernel<C><<<go, C::NTHREADS, C::smem_out>>>(
      d.xn, d.xout, d.p_out_w, d.g_out_w, d.p_out_b, d.g_out_b,
      d.out, rows, D);
}

using tile_cfg = fused::Cfg<64, 64, 32, 2, 2>;   // 64x64 tile, 128 threads

static void trimul_forward(const TriMulBuffers& d, int B, int N, int D,
                           float eps, int outgoing) {
  if (outgoing) trimul_forward<tile_cfg, 1>(d, B, N, D, eps);
  else          trimul_forward<tile_cfg, 0>(d, B, N, D, eps);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    printf("Usage: %s <batch> <seq_len> <hidden_dim> <repeat>\n", argv[0]);
    return 1;
  }
  const int B = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int D = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  if (!valid_problem_size(B, N, D, repeat)) {
    printf("Invalid arguments: <batch>, <seq_len>, <hidden_dim> and <repeat> "
           "must be positive, and all derived sizes must fit their types\n");
    return 1;
  }
  const float eps = 1e-5f;
  const size_t xsz = (size_t)B * N * N * D;
  const size_t msz = (size_t)B * N * N;

  // Checked before anything is allocated, so an unsupported device just exits.
  // BF16 WMMA and cp.async both arrived with Ampere.
  int device = 0;
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  if (props.major < 8) {
    printf("BF16 tensor cores need compute capability 8.0 or newer, "
           "this device is %d.%d\n", props.major, props.minor);
    return 1;
  }

  srand(0);

  float* x = make_random_float(xsz);
  float* mask = (float*)xmalloc(msz * sizeof(float));
  for (size_t i = 0; i < msz; i++) mask[i] = (rand() / (float)RAND_MAX) > 0.1f ? 1.f : 0.f;

  float* norm_in_w = make_random_float(D);
  float* norm_in_b = make_random_float(D);
  float* p_in_w = make_random_float((size_t)2 * D * D);
  float* p_in_b = make_random_float(2 * D);
  float* g_in_w = make_random_float((size_t)2 * D * D);
  float* g_in_b = make_random_float(2 * D);
  float* norm_out_w = make_random_float(D);
  float* norm_out_b = make_random_float(D);
  float* p_out_w = make_random_float((size_t)D * D);
  float* p_out_b = make_random_float(D);
  float* g_out_w = make_random_float((size_t)D * D);
  float* g_out_b = make_random_float(D);
  __nv_bfloat16* p_in_w_bf = make_bfloat16(p_in_w, (size_t)2 * D * D);
  __nv_bfloat16* g_in_w_bf = make_bfloat16(g_in_w, (size_t)2 * D * D);
  __nv_bfloat16* p_out_w_bf = make_bfloat16(p_out_w, (size_t)D * D);
  __nv_bfloat16* g_out_w_bf = make_bfloat16(g_out_w, (size_t)D * D);

  float* ref = (float*)xmalloc(xsz * sizeof(float));

  TriMulBuffers d{};
#define DMALLOC(P, N) CUDA_CHECK(cudaMalloc(&(P), (N) * sizeof(*(P))))
  DMALLOC(d.x, xsz); DMALLOC(d.mask, msz); DMALLOC(d.tri, xsz);
  DMALLOC(d.out, xsz);
  DMALLOC(d.xn, xsz); DMALLOC(d.xout, xsz);
  DMALLOC(d.a, xsz); DMALLOC(d.b, xsz);
  DMALLOC(d.p_in_w, (size_t)2 * D * D); DMALLOC(d.g_in_w, (size_t)2 * D * D);
  DMALLOC(d.p_out_w, (size_t)D * D); DMALLOC(d.g_out_w, (size_t)D * D);
  DMALLOC(d.norm_in_w, D); DMALLOC(d.norm_in_b, D);
  DMALLOC(d.p_in_b, 2 * D); DMALLOC(d.g_in_b, 2 * D);
  DMALLOC(d.norm_out_w, D); DMALLOC(d.norm_out_b, D);
  DMALLOC(d.p_out_b, D); DMALLOC(d.g_out_b, D);
#undef DMALLOC

  // The tensor cores are requested through CUBLAS_GEMM_DEFAULT_TENSOR_OP in
  // the GEMM call itself; cublasSetMathMode(CUBLAS_TENSOR_OP_MATH) has been a
  // deprecated no-op since CUDA 11, and hipBLAS has no equivalent either.
  CUBLAS_CHECK(cublasCreate(&d.blas));

  auto H2D = [](float* dst, const float* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
  };
  auto H2DBF = [](__nv_bfloat16* dst, const __nv_bfloat16* src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
  };
  H2D(d.x, x, xsz); H2D(d.mask, mask, msz);
  H2D(d.norm_in_w, norm_in_w, D); H2D(d.norm_in_b, norm_in_b, D);
  H2DBF(d.p_in_w, p_in_w_bf, (size_t)2 * D * D); H2D(d.p_in_b, p_in_b, 2 * D);
  H2DBF(d.g_in_w, g_in_w_bf, (size_t)2 * D * D); H2D(d.g_in_b, g_in_b, 2 * D);
  H2D(d.norm_out_w, norm_out_w, D); H2D(d.norm_out_b, norm_out_b, D);
  H2DBF(d.p_out_w, p_out_w_bf, (size_t)D * D); H2D(d.p_out_b, p_out_b, D);
  H2DBF(d.g_out_w, g_out_w_bf, (size_t)D * D); H2D(d.g_out_b, g_out_b, D);
  CUDA_CHECK(cudaDeviceSynchronize());

  int errors = 0;

  // Both directions of the triangle projection are benchmarked in turn.
  for (int outgoing = 1; outgoing >= 0; outgoing--) {
    const char* dir = outgoing ? "outgoing" : "incoming";

    trimul_forward_ref(ref, x, mask, norm_in_w, norm_in_b, p_in_w, p_in_b,
                       g_in_w, g_in_b, norm_out_w, norm_out_b, p_out_w, p_out_b,
                       g_out_w, g_out_b, B, N, D, outgoing, eps);

    // host/device correctness check (run once, verify against the reference)
    // before timing; the same pass also warms up the device
    CUDA_CHECK(cudaMemset(d.out, 0, xsz * sizeof(float)));
    trimul_forward(d, B, N, D, eps, outgoing);
    CUDA_CHECK(cudaGetLastError());
    if (!verify_result(d.out, ref, dir, xsz)) errors++;

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++)
      trimul_forward(d, B, N, D, eps, outgoing);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of trimul (%s): %f (us)\n",
           dir, time * 1e-3 / repeat);
  }

  printf("%s\n", errors == 0 ? "PASS" : "FAIL");

  CUBLAS_CHECK(cublasDestroy(d.blas));
  CUDA_CHECK(cudaFree(d.x)); CUDA_CHECK(cudaFree(d.mask));
  CUDA_CHECK(cudaFree(d.tri));
  CUDA_CHECK(cudaFree(d.xn)); CUDA_CHECK(cudaFree(d.xout));
  CUDA_CHECK(cudaFree(d.a)); CUDA_CHECK(cudaFree(d.b));
  CUDA_CHECK(cudaFree(d.out));
  CUDA_CHECK(cudaFree(d.norm_in_w)); CUDA_CHECK(cudaFree(d.norm_in_b));
  CUDA_CHECK(cudaFree(d.p_in_w)); CUDA_CHECK(cudaFree(d.p_in_b));
  CUDA_CHECK(cudaFree(d.g_in_w)); CUDA_CHECK(cudaFree(d.g_in_b));
  CUDA_CHECK(cudaFree(d.norm_out_w)); CUDA_CHECK(cudaFree(d.norm_out_b));
  CUDA_CHECK(cudaFree(d.p_out_w)); CUDA_CHECK(cudaFree(d.p_out_b));
  CUDA_CHECK(cudaFree(d.g_out_w)); CUDA_CHECK(cudaFree(d.g_out_b));

  free(x); free(mask); free(ref);
  free(norm_in_w); free(norm_in_b); free(p_in_w); free(p_in_b);
  free(g_in_w); free(g_in_b); free(norm_out_w); free(norm_out_b);
  free(p_out_w); free(p_out_b); free(g_out_w); free(g_out_b);
  free(p_in_w_bf); free(g_in_w_bf); free(p_out_w_bf); free(g_out_w_bf);
  return 0;
}
