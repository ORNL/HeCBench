/*
 * Triangle Multiplicative Update (TriMul) forward pass -- BF16 HIP port.
 *
 * A key component of the AlphaFold2/3 Evoformer, benchmarked here after
 * NVIDIA cuEquivariance's triangle_multiplicative_update (Apache-2.0).
 * See reference.h for the algorithm and tensor layouts.
 *
 * The math, precision contract, tensor layouts, CLI, and validation match
 * ../trimul-cuda/main.cu: LayerNorm and epilogue arithmetic use FP32, GEMM
 * operands are BF16, matrix products accumulate into FP32, and the triangle
 * projection is a strided-batched BF16 GEMM.
 *
 * The one kernel-level difference is staging: CUDA overlaps its operand
 * staging with cp.async, which has no rocWMMA equivalent here, so this port
 * always takes the synchronous path that CUDA falls back to for partial tiles.
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocwmma/rocwmma.hpp>
#include "reference.h"

#define HIP_CHECK(x) hip_check((x), #x, __FILE__, __LINE__)
static void hip_check(hipError_t e, const char* call, const char* file, int line) {
  if (e != hipSuccess) {
    fprintf(stderr, "HIP error at %s:%d: %s: %s\n",
            file, line, call, hipGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

#define HIPBLAS_CHECK(x) hipblas_check((x), #x, __FILE__, __LINE__)
static void hipblas_check(hipblasStatus_t e, const char* call,
                          const char* file, int line) {
  if (e != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "hipBLAS error at %s:%d: %s: status %d\n",
            file, line, call, (int)e);
    exit(EXIT_FAILURE);
  }
}

static float* make_random_float(size_t n) {
  float* a = (float*)xmalloc(n * sizeof(float));
  for (size_t i = 0; i < n; ++i)
    a[i] = rand() / (float)RAND_MAX * 2.f - 1.f;
  return a;
}

static hip_bfloat16* make_bfloat16(const float* src, size_t n) {
  hip_bfloat16* dst = (hip_bfloat16*)xmalloc(n * sizeof(hip_bfloat16));
  for (size_t i = 0; i < n; ++i) dst[i] = hip_bfloat16(src[i]);
  return dst;
}

// The BF16 operands only keep 8 significant bits, so the result is compared
// against the FP32 reference through error statistics (max abs, max rel, RMS)
// rather than an FP32-tight per-element tolerance.
static bool verify_result(const float* result, const float* ref,
                          const char* name, size_t n) {
  float* h = (float*)xmalloc(n * sizeof(float));
  HIP_CHECK(hipMemcpy(h, result, n * sizeof(float), hipMemcpyDeviceToHost));
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

// Both LayerNorms give a row to a wavefront rather than to a whole block, so
// the reduction stays inside the wavefront: no LDS, no __syncthreads, and no
// idle lanes when D is smaller than a block.  Sum and sum-of-squares are
// accumulated in the same pass and reduced together, which reads the row twice
// instead of three times, and the xor butterfly leaves the totals in every
// lane so no broadcast is needed.
template <int W>
__device__ __forceinline__ void wave_sum2(float& s, float& ss) {
  for (int off = W / 2; off > 0; off >>= 1) {
    s += __shfl_xor(s, off, W);
    ss += __shfl_xor(ss, off, W);
  }
}

template <int TPB, int W>
__global__ void layernorm_contig_bf16_kernel(const float* __restrict__ in,
                                             hip_bfloat16* __restrict__ out,
                                             const float* __restrict__ w,
                                             const float* __restrict__ b,
                                             int rows, int D, float eps) {
  const int lane = threadIdx.x % W;
  const int row = blockIdx.x * (TPB / W) + threadIdx.x / W;
  if (row >= rows) return;
  const float* x = in + (size_t)row * D;
  hip_bfloat16* y = out + (size_t)row * D;

  float s = 0.f, ss = 0.f;
  for (int d = lane; d < D; d += W) {
    const float v = x[d];
    s += v;
    ss += v * v;
  }
  wave_sum2<W>(s, ss);

  const float mean = s / D;
  const float rstd = rsqrtf(ss / D - mean * mean + eps);

  for (int d = lane; d < D; d += W)
    y[d] = hip_bfloat16((x[d] - mean) * rstd * w[d] + b[d]);
}

// Gathering LayerNorm, emitting BF16 directly.
template <int TPB, int W>
__global__ void layernorm_gather_bf16_kernel(const float* __restrict__ tri,
                                             hip_bfloat16* __restrict__ xout,
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
  wave_sum2<W>(s, ss);

  const float mean = s / D;
  const float rstd = rsqrtf(ss / D - mean * mean + eps);

  hip_bfloat16* y = xout + (size_t)row * D;
  for (int d = lane; d < D; d += W)
    y[d] = hip_bfloat16((tri[base + (size_t)d * NN] - mean) * rstd * w[d] + b[d]);
}

// Fused sigmoid-gated dual matrix products, the rocWMMA counterpart of the
// WMMA kernels in ../trimul-cuda/main.cu.  A block owns a BM x BN output tile
// split over WAVES_M x WAVES_N waves; the operand tiles are staged in LDS with
// zero padding so any problem shape is handled, and the accumulators never
// leave the chip before the gated epilogue has consumed them.
namespace fused {

template <int BM_, int BN_, int BK_, int WAVES_M_, int WAVES_N_, int WAVE_>
struct Cfg {
  static constexpr int BM = BM_, BN = BN_, BK = BK_;
  static constexpr int WAVES_M = WAVES_M_, WAVES_N = WAVES_N_;
  static constexpr int NWAVES = WAVES_M * WAVES_N;
  static constexpr int WAVE = WAVE_;
  static constexpr int NTHREADS = NWAVES * WAVE;
  static constexpr int FM = BM / (WAVES_M * 16);
  static constexpr int FN = BN / (WAVES_N * 16);
  static constexpr int LDA = BK + 8;   // bf16 staging row stride
  static constexpr int LDT = 16 + 4;   // fp32 per-wave epilogue tile stride

  static constexpr size_t stage_in  = (size_t)(BM * LDA + 2 * BN * LDA) * sizeof(hip_bfloat16);
  static constexpr size_t stage_out = (size_t)(2 * BM * LDA + 2 * BN * LDA) * sizeof(hip_bfloat16);
  static constexpr size_t epi = (size_t)NWAVES * 2 * 16 * LDT * sizeof(float);

  // Staging buffers per k-step: staging here is synchronous, so one tile is
  // live at a time.  The CUDA port overlaps the copy with the MMAs through
  // cp.async and sets this to two; everything else about the shared-memory
  // layout is the same in both.
  static constexpr int STAGES = 1;
  static constexpr size_t smem_in  = STAGES * stage_in  > epi ? STAGES * stage_in  : epi;
  static constexpr size_t smem_out = STAGES * stage_out > epi ? STAGES * stage_out : epi;

  // Staying within 48 KB keeps the kernels launchable on every BF16-capable
  // device, AMD or NVIDIA, even though an AMD workgroup gets 64 KB of LDS.
  static_assert(smem_in <= 48 * 1024, "input tile exceeds the shared memory limit");
  static_assert(smem_out <= 48 * 1024, "output tile exceeds the shared memory limit");

  static_assert(BM % (WAVES_M * 16) == 0, "BM must tile into WAVES_M*16");
  static_assert(BN % (WAVES_N * 16) == 0, "BN must tile into WAVES_N*16");
  static_assert(BK % 16 == 0, "BK must be a multiple of the rocWMMA k-step");
  static_assert(BK % 8 == 0, "BK must be a multiple of the 16-byte vector width");
};

using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16,
                                rocwmma::bfloat16_t, rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16,
                                rocwmma::bfloat16_t, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

// Stage a ROWS x BK tile of a row-major (*, D) BF16 matrix into LDS,
// zero-padding out-of-range elements.  Uses 16-byte vector copies when the
// alignment allows it, else falls back to scalar.
template <int ROWS, int BK, int LDA, int NTHREADS>
__device__ __forceinline__ void stage_sync(const hip_bfloat16* __restrict__ src,
                                           int ld_src, int row0, int row_limit,
                                           int kt, int D, hip_bfloat16* dst, int tid) {
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
    const hip_bfloat16 zero = hip_bfloat16(0.f);
    for (int idx = tid; idx < ROWS * BK; idx += NTHREADS) {
      const int r = idx / BK, k = idx % BK;
      const int gr = row0 + r, gk = kt + k;
      dst[r * LDA + k] =
          (gr < row_limit && gk < D) ? src[(size_t)gr * ld_src + gk] : zero;
    }
  }
}

}  // namespace fused

// Fused sigmoid-gated dual matrix product (input side), BF16 in / BF16 out.
//
//   proj[m,c] = xn[m,:] . p_in[c,:] + pb[c]
//   gate[m,c] = xn[m,:] . g_in[c,:] + gb[c]
//   v         = sigmoid(gate) * proj * mask[m]
//   c <  D -> a_out[(b*D + c    )*NN + ik]
//   c >= D -> b_out[(b*D + c - D)*NN + ik]     with m = b*NN + ik
//
// Both accumulator sets reuse one A fragment, so proj and gate never reach HBM.
template <typename C>
__global__ __launch_bounds__(C::NTHREADS)
void gated_gemm_in_fused_kernel(const hip_bfloat16* __restrict__ A,
                                const hip_bfloat16* __restrict__ Pw,
                                const hip_bfloat16* __restrict__ Gw,
                                const float* __restrict__ pb,
                                const float* __restrict__ gb,
                                const float* __restrict__ mask,
                                hip_bfloat16* __restrict__ a_out,
                                hip_bfloat16* __restrict__ b_out,
                                int rows, int D, int NN) {
  using fused::stage_sync;
  constexpr int BM = C::BM, BN = C::BN, BK = C::BK, LDA = C::LDA, LDT = C::LDT;
  constexpr int FM = C::FM, FN = C::FN, NT = C::NTHREADS, WAVE = C::WAVE;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  hip_bfloat16* sh = reinterpret_cast<hip_bfloat16*>(smem_raw);

  const int tid = threadIdx.x;
  const int wave = tid / WAVE, lane = tid % WAVE;
  const int wave_m = wave / C::WAVES_N, wave_n = wave % C::WAVES_N;
  const int m0 = blockIdx.y * BM;
  const int c0 = blockIdx.x * BN;
  const int twoD = 2 * D;

  fused::FragAcc accP[FM][FN], accG[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) {
      rocwmma::fill_fragment(accP[i][j], 0.f);
      rocwmma::fill_fragment(accG[i][j], 0.f);
    }

  for (int kt = 0; kt < D; kt += BK) {
    hip_bfloat16* As = sh;
    hip_bfloat16* Ps = sh + BM * LDA;
    hip_bfloat16* Gs = Ps + BN * LDA;
    stage_sync<BM, BK, LDA, NT>(A, D, m0, rows, kt, D, As, tid);
    stage_sync<BN, BK, LDA, NT>(Pw, D, c0, twoD, kt, D, Ps, tid);
    stage_sync<BN, BK, LDA, NT>(Gw, D, c0, twoD, kt, D, Gs, tid);
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      fused::FragA af[FM];
      fused::FragB pf[FN], gf[FN];
#pragma unroll
      for (int i = 0; i < FM; i++)
        rocwmma::load_matrix_sync(af[i], &As[(wave_m * (FM * 16) + i * 16) * LDA + kk], LDA);
#pragma unroll
      for (int j = 0; j < FN; j++) {
        rocwmma::load_matrix_sync(pf[j], &Ps[(wave_n * (FN * 16) + j * 16) * LDA + kk], LDA);
        rocwmma::load_matrix_sync(gf[j], &Gs[(wave_n * (FN * 16) + j * 16) * LDA + kk], LDA);
      }
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) {
          rocwmma::mma_sync(accP[i][j], af[i], pf[j], accP[i][j]);
          rocwmma::mma_sync(accG[i][j], af[i], gf[j], accG[i][j]);
        }
    }
    __syncthreads();
  }

  // Epilogue: one 16x16 tile pair at a time in a per-wave LDS scratch, so
  // proj/gate are combined, gated, masked, cast and scattered on chip.  The
  // staging memory is dead here (all waves passed the final __syncthreads).
  float* Tp = reinterpret_cast<float*>(smem_raw) + wave * (2 * 16 * LDT);
  float* Tg = Tp + 16 * LDT;
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) {
      rocwmma::store_matrix_sync(Tp, accP[i][j], LDT, rocwmma::mem_row_major);
      rocwmma::store_matrix_sync(Tg, accG[i][j], LDT, rocwmma::mem_row_major);
      __builtin_amdgcn_wave_barrier();
      const int mbase = m0 + wave_m * (FM * 16) + i * 16;
      const int cbase = c0 + wave_n * (FN * 16) + j * 16;
      // m fast-varying: consecutive lanes write consecutive ik, which is
      // contiguous in the (B,D,N,N) destination.
      for (int t = lane; t < 256; t += WAVE) {
        const int ml = t & 15, cl = t >> 4;
        const int gm = mbase + ml, gc = cbase + cl;
        if (gm >= rows || gc >= twoD) continue;
        const float proj = Tp[ml * LDT + cl] + pb[gc];
        const float gate = Tg[ml * LDT + cl] + gb[gc];
        const float v = sigmoidf_dev(gate) * proj * mask[gm];
        const int bidx = gm / NN, ik = gm % NN;
        const int d = gc < D ? gc : gc - D;
        hip_bfloat16* dst = gc < D ? a_out : b_out;
        dst[(size_t)(bidx * D + d) * NN + ik] = hip_bfloat16(v);
      }
      __builtin_amdgcn_wave_barrier();
    }
}

// Fused sigmoid-gated dual matrix product (output side), BF16 in / FP32 out.
//
//   out[m,dp] = sigmoid(xn[m,:] . g_out[dp,:] + gb) * (xout[m,:] . p_out[dp,:] + pb)
//
// Unlike the input side the two products have *different* A operands (xn vs
// xout), so two A tiles are staged; the gate/proj combine still stays on-chip.
template <typename C>
__global__ __launch_bounds__(C::NTHREADS)
void gated_gemm_out_fused_kernel(const hip_bfloat16* __restrict__ Xi,
                                 const hip_bfloat16* __restrict__ Xo,
                                 const hip_bfloat16* __restrict__ Pw,
                                 const hip_bfloat16* __restrict__ Gw,
                                 const float* __restrict__ pb,
                                 const float* __restrict__ gb,
                                 float* __restrict__ out, int rows, int D) {
  using fused::stage_sync;
  constexpr int BM = C::BM, BN = C::BN, BK = C::BK, LDA = C::LDA, LDT = C::LDT;
  constexpr int FM = C::FM, FN = C::FN, NT = C::NTHREADS, WAVE = C::WAVE;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  hip_bfloat16* sh = reinterpret_cast<hip_bfloat16*>(smem_raw);

  const int tid = threadIdx.x;
  const int wave = tid / WAVE, lane = tid % WAVE;
  const int wave_m = wave / C::WAVES_N, wave_n = wave % C::WAVES_N;
  const int m0 = blockIdx.y * BM;
  const int c0 = blockIdx.x * BN;

  fused::FragAcc accP[FM][FN], accG[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) {
      rocwmma::fill_fragment(accP[i][j], 0.f);
      rocwmma::fill_fragment(accG[i][j], 0.f);
    }

  for (int kt = 0; kt < D; kt += BK) {
    hip_bfloat16* Xis = sh;
    hip_bfloat16* Xos = sh + BM * LDA;
    hip_bfloat16* Ps  = Xos + BM * LDA;
    hip_bfloat16* Gs  = Ps + BN * LDA;
    stage_sync<BM, BK, LDA, NT>(Xi, D, m0, rows, kt, D, Xis, tid);
    stage_sync<BM, BK, LDA, NT>(Xo, D, m0, rows, kt, D, Xos, tid);
    stage_sync<BN, BK, LDA, NT>(Pw, D, c0, D, kt, D, Ps, tid);
    stage_sync<BN, BK, LDA, NT>(Gw, D, c0, D, kt, D, Gs, tid);
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      fused::FragA xif[FM], xof[FM];
      fused::FragB pf[FN], gf[FN];
#pragma unroll
      for (int i = 0; i < FM; i++) {
        rocwmma::load_matrix_sync(xif[i], &Xis[(wave_m * (FM * 16) + i * 16) * LDA + kk], LDA);
        rocwmma::load_matrix_sync(xof[i], &Xos[(wave_m * (FM * 16) + i * 16) * LDA + kk], LDA);
      }
#pragma unroll
      for (int j = 0; j < FN; j++) {
        rocwmma::load_matrix_sync(pf[j], &Ps[(wave_n * (FN * 16) + j * 16) * LDA + kk], LDA);
        rocwmma::load_matrix_sync(gf[j], &Gs[(wave_n * (FN * 16) + j * 16) * LDA + kk], LDA);
      }
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) {
          rocwmma::mma_sync(accP[i][j], xof[i], pf[j], accP[i][j]);
          rocwmma::mma_sync(accG[i][j], xif[i], gf[j], accG[i][j]);
        }
    }
    __syncthreads();
  }

  float* Tp = reinterpret_cast<float*>(smem_raw) + wave * (2 * 16 * LDT);
  float* Tg = Tp + 16 * LDT;
#pragma unroll
  for (int i = 0; i < FM; i++)
#pragma unroll
    for (int j = 0; j < FN; j++) {
      rocwmma::store_matrix_sync(Tp, accP[i][j], LDT, rocwmma::mem_row_major);
      rocwmma::store_matrix_sync(Tg, accG[i][j], LDT, rocwmma::mem_row_major);
      __builtin_amdgcn_wave_barrier();
      const int mbase = m0 + wave_m * (FM * 16) + i * 16;
      const int cbase = c0 + wave_n * (FN * 16) + j * 16;
      for (int t = lane; t < 256; t += WAVE) {
        const int ml = t >> 4, cl = t & 15;
        const int gm = mbase + ml, gc = cbase + cl;
        if (gm >= rows || gc >= D) continue;
        const float proj = Tp[ml * LDT + cl] + pb[gc];
        const float gate = Tg[ml * LDT + cl] + gb[gc];
        out[(size_t)gm * D + gc] = sigmoidf_dev(gate) * proj;
      }
      __builtin_amdgcn_wave_barrier();
    }
}

struct TriMulBuffers {
  float *x, *mask, *tri, *out;
  hip_bfloat16 *xn, *xout, *a, *b;
  hip_bfloat16 *p_in_w, *g_in_w, *p_out_w, *g_out_w;
  float *norm_in_w, *norm_in_b, *p_in_b, *g_in_b;
  float *norm_out_w, *norm_out_b, *p_out_b, *g_out_b;
  hipblasHandle_t blas;
};

// Triangle projection (BF16 batched GEMM, matrix cores) row-major mapping,
// one N-by-N multiplication for each (batch, channel) plane:
//   outgoing: C[i,j] = sum_k a[i,k] b[j,k]  => Ccm = op(b,T) . op(a,N)
//   incoming: C[i,j] = sum_k a[k,i] b[k,j]  => Ccm = op(b,N) . op(a,T)
template <int OUTGOING>
static void triangle_batched_bf16(const TriMulBuffers& d, int B, int N, int D) {
  const float one = 1.f, zero = 0.f;
  const hipblasOperation_t ta = OUTGOING ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const hipblasOperation_t tb = OUTGOING ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  const hipblasStride stride = (hipblasStride)N * N;
  HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
      d.blas, ta, tb, N, N, N, &one,
      d.b, HIP_R_16BF, N, stride, d.a, HIP_R_16BF, N, stride,
      &zero, d.tri, HIP_R_32F, N, stride, B * D,
      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}

// One TriMul forward pass: LayerNorm -> gated dual product -> triangle
// projection -> LayerNorm -> gated dual product.
template <typename C, int OUTGOING>
static void trimul_forward(const TriMulBuffers& d, int B, int N, int D, float eps) {
  constexpr int TPB = 256;
  constexpr int W = C::WAVE;                  // one wavefront per normalized row
  const int rows = B * N * N;
  const int NN = N * N;
  const int ln_blocks = (rows + TPB / W - 1) / (TPB / W);

  layernorm_contig_bf16_kernel<TPB, W><<<ln_blocks, TPB>>>(
      d.x, d.xn, d.norm_in_w, d.norm_in_b, rows, D, eps);

  const dim3 gi((2 * D + C::BN - 1) / C::BN, (rows + C::BM - 1) / C::BM);
  gated_gemm_in_fused_kernel<C><<<gi, C::NTHREADS, C::smem_in>>>(
      d.xn, d.p_in_w, d.g_in_w, d.p_in_b, d.g_in_b, d.mask,
      d.a, d.b, rows, D, NN);

  triangle_batched_bf16<OUTGOING>(d, B, N, D);

  layernorm_gather_bf16_kernel<TPB, W><<<ln_blocks, TPB>>>(
      d.tri, d.xout, d.norm_out_w, d.norm_out_b, rows, N, D, eps);

  const dim3 go((D + C::BN - 1) / C::BN, (rows + C::BM - 1) / C::BM);
  gated_gemm_out_fused_kernel<C><<<go, C::NTHREADS, C::smem_out>>>(
      d.xn, d.xout, d.p_out_w, d.g_out_w, d.p_out_b, d.g_out_b, d.out, rows, D);
}

// 64x64 tile over 2x2 waves, matching ../trimul-cuda/main.cu.
template <int WAVE>
using tile_cfg = fused::Cfg<64, 64, 32, 2, 2, WAVE>;

static void trimul_forward(const TriMulBuffers& d, int B, int N, int D,
                           float eps, int outgoing, int wave_size) {
  if (wave_size == 64) {
    if (outgoing) trimul_forward<tile_cfg<64>, 1>(d, B, N, D, eps);
    else          trimul_forward<tile_cfg<64>, 0>(d, B, N, D, eps);
  } else {
    if (outgoing) trimul_forward<tile_cfg<32>, 1>(d, B, N, D, eps);
    else          trimul_forward<tile_cfg<32>, 0>(d, B, N, D, eps);
  }
}

int main(int argc, char** argv) {
  if (argc != 5) {
    printf("Usage: %s <batch> <seq_len> <hidden_dim> <repeat>\n", argv[0]);
    return 1;
  }
  const int B = atoi(argv[1]), N = atoi(argv[2]), D = atoi(argv[3]);
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
  int device = 0;
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  const int wave_size = props.warpSize;
  if (wave_size != 32 && wave_size != 64) {
    printf("rocWMMA needs a 32- or 64-wide wave, this device has %d\n", wave_size);
    return 1;
  }

  srand(0);
  float* x = make_random_float(xsz);
  float* mask = (float*)xmalloc(msz * sizeof(float));
  for (size_t i = 0; i < msz; ++i)
    mask[i] = rand() / (float)RAND_MAX > 0.1f ? 1.f : 0.f;
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
  hip_bfloat16* p_in_w_bf = make_bfloat16(p_in_w, (size_t)2 * D * D);
  hip_bfloat16* g_in_w_bf = make_bfloat16(g_in_w, (size_t)2 * D * D);
  hip_bfloat16* p_out_w_bf = make_bfloat16(p_out_w, (size_t)D * D);
  hip_bfloat16* g_out_w_bf = make_bfloat16(g_out_w, (size_t)D * D);
  float* ref = (float*)xmalloc(xsz * sizeof(float));

  TriMulBuffers d{};
#define DMALLOC(P, N) HIP_CHECK(hipMalloc(&(P), (N) * sizeof(*(P))))
  DMALLOC(d.x, xsz); DMALLOC(d.mask, msz); DMALLOC(d.tri, xsz);
  DMALLOC(d.out, xsz);
  DMALLOC(d.xn, xsz); DMALLOC(d.xout, xsz); DMALLOC(d.a, xsz); DMALLOC(d.b, xsz);
  DMALLOC(d.p_in_w, (size_t)2 * D * D); DMALLOC(d.g_in_w, (size_t)2 * D * D);
  DMALLOC(d.p_out_w, (size_t)D * D); DMALLOC(d.g_out_w, (size_t)D * D);
  DMALLOC(d.norm_in_w, D); DMALLOC(d.norm_in_b, D);
  DMALLOC(d.p_in_b, 2 * D); DMALLOC(d.g_in_b, 2 * D);
  DMALLOC(d.norm_out_w, D); DMALLOC(d.norm_out_b, D);
  DMALLOC(d.p_out_b, D); DMALLOC(d.g_out_b, D);
#undef DMALLOC
  HIPBLAS_CHECK(hipblasCreate(&d.blas));

  auto H2D = [](float* dst, const float* src, size_t n) {
    HIP_CHECK(hipMemcpy(dst, src, n * sizeof(float), hipMemcpyHostToDevice));
  };
  auto H2DBF = [](hip_bfloat16* dst, const hip_bfloat16* src, size_t n) {
    HIP_CHECK(hipMemcpy(dst, src, n * sizeof(hip_bfloat16),
                        hipMemcpyHostToDevice));
  };
  H2D(d.x, x, xsz); H2D(d.mask, mask, msz);
  H2D(d.norm_in_w, norm_in_w, D); H2D(d.norm_in_b, norm_in_b, D);
  H2DBF(d.p_in_w, p_in_w_bf, (size_t)2 * D * D); H2D(d.p_in_b, p_in_b, 2 * D);
  H2DBF(d.g_in_w, g_in_w_bf, (size_t)2 * D * D); H2D(d.g_in_b, g_in_b, 2 * D);
  H2D(d.norm_out_w, norm_out_w, D); H2D(d.norm_out_b, norm_out_b, D);
  H2DBF(d.p_out_w, p_out_w_bf, (size_t)D * D); H2D(d.p_out_b, p_out_b, D);
  H2DBF(d.g_out_w, g_out_w_bf, (size_t)D * D); H2D(d.g_out_b, g_out_b, D);
  HIP_CHECK(hipDeviceSynchronize());

  int errors = 0;

  // Both directions of the triangle projection are benchmarked in turn.
  for (int outgoing = 1; outgoing >= 0; --outgoing) {
    const char* dir = outgoing ? "outgoing" : "incoming";

    trimul_forward_ref(ref, x, mask, norm_in_w, norm_in_b, p_in_w, p_in_b,
                       g_in_w, g_in_b, norm_out_w, norm_out_b, p_out_w, p_out_b,
                       g_out_w, g_out_b, B, N, D, outgoing, eps);

    // host/device correctness check (run once, verify against the reference)
    // before timing; the same pass also warms up the device
    HIP_CHECK(hipMemset(d.out, 0, xsz * sizeof(float)));
    trimul_forward(d, B, N, D, eps, outgoing, wave_size);
    HIP_CHECK(hipGetLastError());
    if (!verify_result(d.out, ref, dir, xsz)) errors++;

    const auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; ++r)
      trimul_forward(d, B, N, D, eps, outgoing, wave_size);

    HIP_CHECK(hipDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    const auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of trimul (%s): %f (us)\n",
           dir, time * 1e-3 / repeat);
  }

  printf("%s\n", errors == 0 ? "PASS" : "FAIL");

  HIPBLAS_CHECK(hipblasDestroy(d.blas));
  HIP_CHECK(hipFree(d.x)); HIP_CHECK(hipFree(d.mask));
  HIP_CHECK(hipFree(d.tri)); HIP_CHECK(hipFree(d.out));
  HIP_CHECK(hipFree(d.xn)); HIP_CHECK(hipFree(d.xout));
  HIP_CHECK(hipFree(d.a)); HIP_CHECK(hipFree(d.b));
  HIP_CHECK(hipFree(d.p_in_w)); HIP_CHECK(hipFree(d.g_in_w));
  HIP_CHECK(hipFree(d.p_out_w)); HIP_CHECK(hipFree(d.g_out_w));
  HIP_CHECK(hipFree(d.norm_in_w)); HIP_CHECK(hipFree(d.norm_in_b));
  HIP_CHECK(hipFree(d.p_in_b)); HIP_CHECK(hipFree(d.g_in_b));
  HIP_CHECK(hipFree(d.norm_out_w)); HIP_CHECK(hipFree(d.norm_out_b));
  HIP_CHECK(hipFree(d.p_out_b)); HIP_CHECK(hipFree(d.g_out_b));

  free(x); free(mask); free(ref);
  free(norm_in_w); free(norm_in_b); free(p_in_w); free(p_in_b);
  free(g_in_w); free(g_in_b); free(norm_out_w); free(norm_out_b);
  free(p_out_w); free(p_out_b); free(g_out_w); free(g_out_b);
  free(p_in_w_bf); free(g_in_w_bf); free(p_out_w_bf); free(g_out_w_bf);
  return 0;
}
