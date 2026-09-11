/*
 * Triangle Multiplicative Update (TriMul) forward pass -- BF16 SYCL port.
 *
 * A key component of the AlphaFold2/3 Evoformer, benchmarked here after
 * NVIDIA cuEquivariance's triangle_multiplicative_update (Apache-2.0).
 * See reference.h for the algorithm and tensor layouts.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#ifdef USE_ONEMKL
#include <oneapi/mkl.hpp>
#endif
#include "reference.h"

using sycl::ext::oneapi::bfloat16;

static float* make_random_float(size_t n) {
  float* a = (float*)xmalloc(n * sizeof(float));
  for (size_t i = 0; i < n; i++) a[i] = rand() / (float)RAND_MAX * 2.f - 1.f;
  return a;
}

static bfloat16* make_bfloat16(const float* src, size_t n) {
  bfloat16* dst = (bfloat16*)xmalloc(n * sizeof(bfloat16));
  for (size_t i = 0; i < n; i++) dst[i] = bfloat16(src[i]);
  return dst;
}

// The BF16 operands only keep 8 significant bits, so the result is compared
// against the FP32 reference through error statistics (max abs, max rel, RMS)
// rather than an FP32-tight per-element tolerance.
static bool verify_result(sycl::queue& q, const float* d_result,
                          const float* ref, const char* name, size_t n) {
  float* h = (float*)xmalloc(n * sizeof(float));
  q.memcpy(h, d_result, n * sizeof(float)).wait();
  double max_abs = 0, max_rel = 0, max_scaled = 0, sse = 0, sref = 0;
  for (size_t i = 0; i < n; i++) {
    const double e = fabs((double)ref[i] - (double)h[i]);
    max_abs = e > max_abs ? e : max_abs;
    const double rel = e / (fabs((double)ref[i]) + 1e-6);
    max_rel = rel > max_rel ? rel : max_rel;
    const double scaled = e / (1.0 + fabs((double)ref[i]));
    max_scaled = scaled > max_scaled ? scaled : max_scaled;
    sse += e * e;
    sref += (double)ref[i] * (double)ref[i];
  }
  free(h);
  const double rms_rel = sqrt(sse / (sref + 1e-12));
  printf("%s: max_abs=%.4g  max_rel=%.4g  max_scaled=%.4g  "
         "rms_rel=%.4g (tolerance 5e-2)\n",
         name, max_abs, max_rel, max_scaled, rms_rel);
  return max_scaled < 5e-2 && rms_rel < 5e-2;
}

static inline float sigmoidf_dev(float v) {
  if (v >= 0.f) return 1.0f / (1.0f + sycl::exp(-v));
  const float e = sycl::exp(v);
  return e / (1.0f + e);
}

// ---------------------------------------------------------------------------
// 1. LayerNorm over D for contiguous rows, emitting BF16 directly so the
//    normalized activations never round-trip through FP32 memory.
//    in: (rows, D) FP32   out: (rows, D) BF16
// ---------------------------------------------------------------------------
// Both LayerNorms give a row to a sub-group rather than to a whole work-group,
// so the reduction stays inside the sub-group: no local memory, no barriers,
// and no idle work-items when D is smaller than a work-group.  Sum and
// sum-of-squares are accumulated in the same pass and reduced together, which
// reads the row twice instead of three times, and reduce_over_group leaves the
// totals in every work-item so no broadcast is needed.
template <int TPB>
static void layernorm_contig_bf16(sycl::queue& q, int rows, const float* in,
                                  bfloat16* out, const float* w, const float* b,
                                  int D, float eps, int sg_size) {
  const int rpb = TPB / sg_size;              // rows per work-group
  const size_t groups = ((size_t)rows + rpb - 1) / rpb;
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * TPB),
                                     sycl::range<1>(TPB)),
                   [=](sycl::nd_item<1> item) [[sycl::reqd_work_group_size(TPB)]] {
      auto sg = item.get_sub_group();
      const int W = sg.get_local_linear_range();
      const int lane = sg.get_local_linear_id();
      const size_t row = item.get_group(0) * (TPB / W) + sg.get_group_linear_id();
      if (row >= (size_t)rows) return;
      const float* x = in + row * D;

      float s = 0.f, ss = 0.f;
      for (int d = lane; d < D; d += W) {
        const float v = x[d];
        s += v;
        ss += v * v;
      }
      s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
      ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());

      const float mean = s / D;
      const float rstd = sycl::rsqrt(ss / D - mean * mean + eps);

      for (int d = lane; d < D; d += W)
        out[row * D + d] = bfloat16((x[d] - mean) * rstd * w[d] + b[d]);
    });
  });
}

// ---------------------------------------------------------------------------
// 4. LayerNorm over D reading strided channel tri[(b*D+d)*N*N + i*N+j] and
//    storing to contiguous xout[(b*N+i)*N+j, D], also emitting BF16 directly.
// ---------------------------------------------------------------------------
template <int TPB>
static void layernorm_gather_bf16(sycl::queue& q, int rows, const float* tri,
                                  bfloat16* xout,
                                  const float* w, const float* b, int N, int D,
                                  float eps, int sg_size) {
  const int rpb = TPB / sg_size;              // rows per work-group
  const size_t groups = ((size_t)rows + rpb - 1) / rpb;
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * TPB),
                                     sycl::range<1>(TPB)),
                   [=](sycl::nd_item<1> item) [[sycl::reqd_work_group_size(TPB)]] {
      auto sg = item.get_sub_group();
      const int W = sg.get_local_linear_range();
      const int lane = sg.get_local_linear_id();
      const size_t row = item.get_group(0) * (TPB / W) + sg.get_group_linear_id();
      if (row >= (size_t)rows) return;
      const int NN = N * N;
      // row = (b*N+i)*N + j, and the channel d is strided by N*N
      const size_t base = (size_t)(row / NN) * D * NN + (row % NN);

      float s = 0.f, ss = 0.f;
      for (int d = lane; d < D; d += W) {
        const float v = tri[base + (size_t)d * NN];
        s += v;
        ss += v * v;
      }
      s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
      ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());

      const float mean = s / D;
      const float rstd = sycl::rsqrt(ss / D - mean * mean + eps);

      bfloat16* y = xout + row * D;
      for (int d = lane; d < D; d += W)
        y[d] = bfloat16((tri[base + (size_t)d * NN] - mean) * rstd * w[d] + b[d]);
    });
  });
}

// ---------------------------------------------------------------------------
// A work-group owns a BM x BN output tile split over SGS_M x SGS_N
// sub-groups; the operand tiles are staged in local memory with zero padding
// so any problem shape is handled, and the accumulators never leave the chip
// before the gated epilogue has consumed them.
// TODO CUDA overlaps its staging with cp.async
// ---------------------------------------------------------------------------
using namespace sycl::ext::oneapi::experimental::matrix;

namespace fused {

constexpr int BM = 64, BN = 64, BK = 32;   // work-group tile
constexpr int SGS_M = 2, SGS_N = 2;        // sub-groups per tile
constexpr int NSG = SGS_M * SGS_N;
constexpr int FM = BM / (SGS_M * 16);      // fragments per sub-group
constexpr int FN = BN / (SGS_N * 16);
constexpr int LDA = BK + 8;                // bf16 staging row stride
constexpr int LDT = 16 + 4;                // fp32 epilogue tile stride
#ifndef USE_ONEMKL
constexpr int LDM = BM + 8, LDN = BN + 8;  // k-major staging row strides
#endif

// One and two staged A tiles, for the input and the output side.  CUDA lets
// its epilogue scratch alias the staging memory; local accessors cannot alias,
// so the scratch is a separate allocation that keeps the total under the 48 KB
// every BF16-capable device provides.
constexpr size_t stage_1a = (size_t)BM * LDA + 2 * (size_t)BN * LDA;
constexpr size_t stage_2a = 2 * (size_t)BM * LDA + 2 * (size_t)BN * LDA;
constexpr size_t epi_elems = (size_t)NSG * 2 * 16 * LDT;

static_assert(BM % (SGS_M * 16) == 0, "BM must tile into SGS_M*16");
static_assert(BN % (SGS_N * 16) == 0, "BN must tile into SGS_N*16");
static_assert(BK % 16 == 0, "BK must be a multiple of the joint_matrix k-step");
static_assert(BK % 8 == 0, "BK must be a multiple of the 16-byte vector width");
static_assert((stage_2a * sizeof(bfloat16) + epi_elems * sizeof(float)) <= 48 * 1024,
              "the staged tiles exceed the local memory every device provides");

using LocalBF = sycl::local_accessor<bfloat16, 1>;
using LocalF = sycl::local_accessor<float, 1>;
using BFPtr = sycl::multi_ptr<bfloat16, sycl::access::address_space::local_space,
                              sycl::access::decorated::no>;

// A work-group is NSG sub-groups wide, so the width the device runs the
// kernels at sets the work-group size.  The host picks the widest width the
// device reports and sizes the launch and the per-sub-group scratch for
// exactly NSG sub-groups of it.  The width is left for the device to choose
// rather than fixed with reqd_sub_group_size, so the kernels take it on trust
// that the device runs them at the width it advertised: a narrower one would
// split the work-group into more sub-groups than the tiling and the scratch
// are sized for.

// Does the device advertise bf16(A) x bf16(B) -> fp32(C/D) at 16x16x16?
// An empty list means the backend does not implement the query rather than
// that the hardware lacks the combination, so the kernels are given a try and
// an unsupported device reports itself through kernel_not_supported instead.
static bool device_supports(const sycl::device& dev) {
  const auto combos =
      dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
  if (combos.empty()) return true;
  for (const auto& c : combos) {
    if (c.atype != matrix_type::bf16 || c.btype != matrix_type::bf16 ||
        c.ctype != matrix_type::fp32 || c.dtype != matrix_type::fp32)
      continue;
    const bool m_ok = (c.msize == 16) || (c.msize == 0 && 16 <= c.max_msize);
    const bool n_ok = (c.nsize == 16) || (c.nsize == 0 && 16 <= c.max_nsize);
    const bool k_ok = (c.ksize == 16) || (c.ksize == 0 && 16 <= c.max_ksize);
    if (m_ok && n_ok && k_ok) return true;
  }
  return false;
}

// Stage a ROWS x BK tile of a row-major (*, ld_src) BF16 matrix into local
// memory, zero-padding out-of-range elements.  Uses 16-byte vector copies
// when the alignment allows it, else falls back to scalar.
template <int ROWS>
static inline void stage_sync(const bfloat16* src, int ld_src, int row0,
                              int row_limit, int kt, int klimit,
                              BFPtr dst, int tid, int nthreads) {
  constexpr int VW = 8;              // bf16 per 16-byte vector
  constexpr int VPR = BK / VW;
  using vec16 = sycl::vec<uint32_t, 4>;
  // The alignment of every vector load depends on the source pitch, not on
  // the k-limit; a partial row tile still takes this path and zero-fills the
  // out-of-range rows below.
  if ((ld_src % VW) == 0 && kt + BK <= klimit) {
    for (int idx = tid; idx < ROWS * VPR; idx += nthreads) {
      const int r = idx / VPR, v = idx % VPR;
      const int gr = row0 + r;
      vec16 val(0u);
      if (gr < row_limit)
        val = *reinterpret_cast<const vec16*>(src + (size_t)gr * ld_src + kt + v * VW);
      *reinterpret_cast<vec16*>(&dst[r * LDA + v * VW]) = val;
    }
  } else {
    for (int idx = tid; idx < ROWS * BK; idx += nthreads) {
      const int r = idx / BK, k = idx % BK;
      const int gr = row0 + r, gk = kt + k;
      dst[r * LDA + k] = (gr < row_limit && gk < klimit)
                             ? src[(size_t)gr * ld_src + gk] : bfloat16(0.f);
    }
  }
}

// Stage a BK x COLS tile of a row-major (*, ld_src) BF16 matrix into local
// memory keeping the k index as the slow axis, so that a column-major operand
// is staged without transposing anything: the copies stay contiguous and the
// fragment load flips its layout instead.
template <int COLS, int LDX>
static inline void stage_sync_k(const bfloat16* src, int ld_src, int col0,
                                int col_limit, int kt, int klimit,
                                BFPtr dst, int tid, int nthreads) {
  constexpr int VW = 8;
  constexpr int VPR = COLS / VW;
  using vec16 = sycl::vec<uint32_t, 4>;
  // Unlike stage_sync, the bounded axis here is also the vectorized one, so a
  // vector can straddle col_limit and a partial tile has to go scalar.
  if ((ld_src % VW) == 0 && (col0 % VW) == 0 && col0 + COLS <= col_limit &&
      kt + BK <= klimit) {
    for (int idx = tid; idx < BK * VPR; idx += nthreads) {
      const int k = idx / VPR, v = idx % VPR;
      const bfloat16* s = src + (size_t)(kt + k) * ld_src + col0 + v * VW;
      bfloat16* t = &dst[k * LDX + v * VW];
      *reinterpret_cast<vec16*>(t) = *reinterpret_cast<const vec16*>(s);
    }
  } else {
    for (int idx = tid; idx < BK * COLS; idx += nthreads) {
      const int k = idx / COLS, c = idx % COLS;
      const int gc = col0 + c, gk = kt + k;
      dst[k * LDX + c] = (gc < col_limit && gk < klimit)
                             ? src[(size_t)gk * ld_src + gc] : bfloat16(0.f);
    }
  }
}

}  // namespace fused

// 2. Fused sigmoid-gated dual matrix product (input side), BF16 in / BF16 out.
//
//   proj[m,c] = xn[m,:] . p_in[c,:] + pb[c]
//   gate[m,c] = xn[m,:] . g_in[c,:] + gb[c]
//   v         = sigmoid(gate) * proj * mask[m]
//   c <  D -> a[(b*D + c    )*NN + ik]
//   c >= D -> b[(b*D + c - D)*NN + ik]     with m = b*NN + ik
//
// Both accumulator sets reuse one A fragment, so proj and gate never reach
// global memory.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wincorrect-sub-group-size"
template <int SG>
static void gated_gemm_in_fused(sycl::queue& q, int rows,
                          const bfloat16* xn, const float* mask,
                          const bfloat16* p_in_w, const float* p_in_b,
                          const bfloat16* g_in_w, const float* g_in_b,
                          bfloat16* a, bfloat16* bb, int B, int N, int D) {
  using namespace fused;
  const int twoD = 2 * D;
  constexpr int nthreads = NSG * SG;
  sycl::range<2> gws((size_t)((rows + BM - 1) / BM),
                     (size_t)((twoD + BN - 1) / BN) * nthreads);
  sycl::range<2> lws(1, (size_t)nthreads);
  q.submit([&](sycl::handler& h) {
    LocalBF sh(sycl::range<1>(stage_1a), h);
    LocalF epi(sycl::range<1>(epi_elems), h);
    h.parallel_for(sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> item)
                   [[sycl::reqd_work_group_size(1, nthreads),
                     sycl::reqd_sub_group_size(SG)]] {
      auto grp = item.get_group();
      auto sg = item.get_sub_group();
      const int NN = N * N;
      const int tid = item.get_local_id(1);
      const int sgid = sg.get_group_linear_id();
      const int lane = sg.get_local_linear_id();
      const int sgw = sg.get_local_linear_range();
      const int sg_m = sgid / SGS_N, sg_n = sgid % SGS_N;
      const int m0 = grp.get_group_id(0) * BM;
      const int c0 = grp.get_group_id(1) * BN;

      auto base = sh.get_multi_ptr<sycl::access::decorated::no>();
      auto As = base, Ps = base + BM * LDA, Gs = base + BM * LDA + BN * LDA;

      joint_matrix<sycl::sub_group, float, use::accumulator, 16, 16> accP[FM][FN], accG[FM][FN];
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) {
          joint_matrix_fill(sg, accP[i][j], 0.f);
          joint_matrix_fill(sg, accG[i][j], 0.f);
        }

      // the weights are (2D, D) row-major, i.e. column-major B with stride D
      for (int kt = 0; kt < D; kt += BK) {
        stage_sync<BM>(xn, D, m0, rows, kt, D, As, tid, nthreads);
        stage_sync<BN>(p_in_w, D, c0, twoD, kt, D, Ps, tid, nthreads);
        stage_sync<BN>(g_in_w, D, c0, twoD, kt, D, Gs, tid, nthreads);
        sycl::group_barrier(grp);

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
          joint_matrix<sycl::sub_group, bfloat16, use::a, 16, 16, layout::row_major> af[FM];
          joint_matrix<sycl::sub_group, bfloat16, use::b, 16, 16, layout::col_major> pf[FN], gf[FN];
#pragma unroll
          for (int i = 0; i < FM; i++)
            joint_matrix_load(sg, af[i], As + (sg_m * (FM * 16) + i * 16) * LDA + kk, LDA);
#pragma unroll
          for (int j = 0; j < FN; j++) {
            joint_matrix_load(sg, pf[j], Ps + (sg_n * (FN * 16) + j * 16) * LDA + kk, LDA);
            joint_matrix_load(sg, gf[j], Gs + (sg_n * (FN * 16) + j * 16) * LDA + kk, LDA);
          }
#pragma unroll
          for (int i = 0; i < FM; i++)
#pragma unroll
            for (int j = 0; j < FN; j++) {
              joint_matrix_mad(sg, accP[i][j], af[i], pf[j], accP[i][j]);
              joint_matrix_mad(sg, accG[i][j], af[i], gf[j], accG[i][j]);
            }
        }
        sycl::group_barrier(grp);
      }

      // Epilogue: one 16x16 tile pair at a time in a per-sub-group scratch, so
      // proj/gate are combined, gated, masked, cast and scattered on chip.
      auto Tp = epi.get_multi_ptr<sycl::access::decorated::no>() + sgid * (2 * 16 * LDT);
      auto Tg = Tp + 16 * LDT;
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) {
          joint_matrix_store(sg, accP[i][j], Tp, LDT, layout::row_major);
          joint_matrix_store(sg, accG[i][j], Tg, LDT, layout::row_major);
          sycl::group_barrier(sg);
          const int mbase = m0 + sg_m * (FM * 16) + i * 16;
          const int cbase = c0 + sg_n * (FN * 16) + j * 16;
          // m fast-varying: consecutive lanes write consecutive ik, which is
          // contiguous in the (B,D,N,N) destination.
          for (int t = lane; t < 256; t += sgw) {
            const int ml = t & 15, cl = t >> 4;
            const int gm = mbase + ml, gc = cbase + cl;
            if (gm >= rows || gc >= twoD) continue;
            const float proj = Tp[ml * LDT + cl] + p_in_b[gc];
            const float gate = Tg[ml * LDT + cl] + g_in_b[gc];
            const float v = sigmoidf_dev(gate) * proj * mask[gm];
            const int bidx = gm / NN, ik = gm % NN;
            const int dch = gc < D ? gc : gc - D;
            bfloat16* dst = gc < D ? a : bb;
            dst[(size_t)(bidx * D + dch) * NN + ik] = bfloat16(v);
          }
          sycl::group_barrier(sg);
        }
    });
  });
}

// 3. Triangle projection.
//   outgoing: C[i,j] = sum_k a[i,k] * b[j,k]
//   incoming: C[i,j] = sum_k a[k,i] * b[k,j]
#ifdef USE_ONEMKL
// TODO The cuBLAS and rocBLAS backends of oneMKL Interfaces instantiate
// gemm_batch that may lack bf16, so this path may need a device
// whose backend provides the bf16 form.
template <int OUTGOING, int SG>
static void triangle_batched_bf16(sycl::queue& q, const bfloat16* a,
                            const bfloat16* bb, float* tri, int N, int BD) {
  namespace mkl = oneapi::mkl;
  const std::int64_t plane_sz = (std::int64_t)N * N;
  const auto ta = OUTGOING ? mkl::transpose::trans : mkl::transpose::nontrans;
  const auto tb = OUTGOING ? mkl::transpose::nontrans : mkl::transpose::trans;
  mkl::blas::column_major::gemm_batch(
      q, ta, tb, N, N, N, 1.f,
      bb, N, plane_sz,
      a, N, plane_sz,
      0.f, tri, N, plane_sz, BD);
}
#else
// Without a BLAS the stage is the same staged tile kernel as the other GEMMs,
// one (b,d) plane per work-group row of the grid.
template <int OUTGOING, int SG>
static void triangle_batched_bf16(sycl::queue& q, const bfloat16* a,
                            const bfloat16* bb, float* tri, int N, int BD) {
  using namespace fused;
  constexpr int nthreads = NSG * SG;
  const int tiles_m = (N + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  sycl::range<3> gws((size_t)BD, (size_t)tiles_m, (size_t)tiles_n * nthreads);
  sycl::range<3> lws(1, 1, nthreads);
  q.submit([&](sycl::handler& h) {
    constexpr size_t tri_stage = OUTGOING ? (size_t)BM * LDA + (size_t)BN * LDA
                                          : (size_t)BK * LDM + (size_t)BK * LDN;
    LocalBF sh(sycl::range<1>(tri_stage), h);
    LocalF epi(sycl::range<1>((size_t)NSG * 16 * LDT), h);
    h.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item)
                   [[sycl::reqd_work_group_size(1, 1, nthreads),
                     sycl::reqd_sub_group_size(SG)]] {
      auto grp = item.get_group();
      auto sg = item.get_sub_group();
      const size_t plane = grp.get_group_id(0) * N * N;
      const bfloat16* A = a + plane;
      const bfloat16* Bp = bb + plane;
      float* C = tri + plane;

      const int tid = item.get_local_id(2);
      const int sgid = sg.get_group_linear_id();
      const int lane = sg.get_local_linear_id();
      const int sgw = sg.get_local_linear_range();
      const int sg_m = sgid / SGS_N, sg_n = sgid % SGS_N;
      const int m0 = grp.get_group_id(1) * BM;
      const int n0 = grp.get_group_id(2) * BN;

      auto base = sh.get_multi_ptr<sycl::access::decorated::no>();
      auto As = base, Bs = base + (OUTGOING ? BM * LDA : BK * LDM);

      joint_matrix<sycl::sub_group, float, use::accumulator, 16, 16> acc[FM][FN];
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) joint_matrix_fill(sg, acc[i][j], 0.f);

      for (int kt = 0; kt < N; kt += BK) {
        if constexpr (OUTGOING) {
          stage_sync<BM>(A, N, m0, N, kt, N, As, tid, nthreads);
          stage_sync<BN>(Bp, N, n0, N, kt, N, Bs, tid, nthreads);
        } else {
          stage_sync_k<BM, LDM>(A, N, m0, N, kt, N, As, tid, nthreads);
          stage_sync_k<BN, LDN>(Bp, N, n0, N, kt, N, Bs, tid, nthreads);
        }
        sycl::group_barrier(grp);

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
          if constexpr (OUTGOING) {
            joint_matrix<sycl::sub_group, bfloat16, use::a, 16, 16, layout::row_major> af[FM];
            joint_matrix<sycl::sub_group, bfloat16, use::b, 16, 16, layout::col_major> bf[FN];
#pragma unroll
            for (int i = 0; i < FM; i++)
              joint_matrix_load(sg, af[i], As + (sg_m * (FM * 16) + i * 16) * LDA + kk, LDA);
#pragma unroll
            for (int j = 0; j < FN; j++)
              joint_matrix_load(sg, bf[j], Bs + (sg_n * (FN * 16) + j * 16) * LDA + kk, LDA);
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
              for (int j = 0; j < FN; j++)
                joint_matrix_mad(sg, acc[i][j], af[i], bf[j], acc[i][j]);
          } else {
            joint_matrix<sycl::sub_group, bfloat16, use::a, 16, 16, layout::col_major> af[FM];
            joint_matrix<sycl::sub_group, bfloat16, use::b, 16, 16, layout::row_major> bf[FN];
#pragma unroll
            for (int i = 0; i < FM; i++)
              joint_matrix_load(sg, af[i], As + kk * LDM + sg_m * (FM * 16) + i * 16, LDM);
#pragma unroll
            for (int j = 0; j < FN; j++)
              joint_matrix_load(sg, bf[j], Bs + kk * LDN + sg_n * (FN * 16) + j * 16, LDN);
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
              for (int j = 0; j < FN; j++)
                joint_matrix_mad(sg, acc[i][j], af[i], bf[j], acc[i][j]);
          }
        }
        sycl::group_barrier(grp);
      }

      auto T = epi.get_multi_ptr<sycl::access::decorated::no>() + sgid * (16 * LDT);
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) {
          joint_matrix_store(sg, acc[i][j], T, LDT, layout::row_major);
          sycl::group_barrier(sg);
          const int mbase = m0 + sg_m * (FM * 16) + i * 16;
          const int nbase = n0 + sg_n * (FN * 16) + j * 16;
          for (int t = lane; t < 256; t += sgw) {
            const int ml = t >> 4, nl = t & 15;
            const int gi = mbase + ml, gj = nbase + nl;
            if (gi < N && gj < N) C[(size_t)gi * N + gj] = T[ml * LDT + nl];
          }
          sycl::group_barrier(sg);
        }
    });
  });
}
#endif  // USE_ONEMKL

// 5. Fused sigmoid-gated dual matrix product (output side), BF16 in / FP32 out.
//
//   out[m,dp] = sigmoid(xn[m,:] . g_out[dp,:] + gb) * (xout[m,:] . p_out[dp,:] + pb)
//
// Unlike the input side the two products have *different* A operands (xn vs
// xout), so two A tiles are staged; the gate/proj combine still stays on-chip.
template <int SG>
static void gated_gemm_out_fused(sycl::queue& q, int rows,
                           const bfloat16* Xi, const bfloat16* Xo,
                           const bfloat16* p_out_w, const float* p_out_b,
                           const bfloat16* g_out_w, const float* g_out_b,
                           float* out, int D) {
  using namespace fused;
  constexpr int nthreads = NSG * SG;
  sycl::range<2> gws((size_t)((rows + BM - 1) / BM),
                     (size_t)((D + BN - 1) / BN) * nthreads);
  sycl::range<2> lws(1, nthreads);
  q.submit([&](sycl::handler& h) {
    LocalBF sh(sycl::range<1>(stage_2a), h);
    LocalF epi(sycl::range<1>(epi_elems), h);
    h.parallel_for(sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> item)
                   [[sycl::reqd_work_group_size(1, nthreads),
                     sycl::reqd_sub_group_size(SG)]] {
      auto grp = item.get_group();
      auto sg = item.get_sub_group();
      const int tid = item.get_local_id(1);
      const int sgid = sg.get_group_linear_id();
      const int lane = sg.get_local_linear_id();
      const int sgw = sg.get_local_linear_range();
      const int sg_m = sgid / SGS_N, sg_n = sgid % SGS_N;
      const int m0 = grp.get_group_id(0) * BM;
      const int c0 = grp.get_group_id(1) * BN;

      auto base = sh.get_multi_ptr<sycl::access::decorated::no>();
      auto Xis = base, Xos = base + BM * LDA;
      auto Ps = Xos + BM * LDA, Gs = Ps + BN * LDA;

      joint_matrix<sycl::sub_group, float, use::accumulator, 16, 16> accP[FM][FN], accG[FM][FN];
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) {
          joint_matrix_fill(sg, accP[i][j], 0.f);
          joint_matrix_fill(sg, accG[i][j], 0.f);
        }

      // the gate reads the normalized input, the projection the triangle output
      for (int kt = 0; kt < D; kt += BK) {
        stage_sync<BM>(Xi, D, m0, rows, kt, D, Xis, tid, nthreads);
        stage_sync<BM>(Xo, D, m0, rows, kt, D, Xos, tid, nthreads);
        stage_sync<BN>(p_out_w, D, c0, D, kt, D, Ps, tid, nthreads);
        stage_sync<BN>(g_out_w, D, c0, D, kt, D, Gs, tid, nthreads);
        sycl::group_barrier(grp);

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
          joint_matrix<sycl::sub_group, bfloat16, use::a, 16, 16, layout::row_major> xif[FM], xof[FM];
          joint_matrix<sycl::sub_group, bfloat16, use::b, 16, 16, layout::col_major> pf[FN], gf[FN];
#pragma unroll
          for (int i = 0; i < FM; i++) {
            joint_matrix_load(sg, xif[i], Xis + (sg_m * (FM * 16) + i * 16) * LDA + kk, LDA);
            joint_matrix_load(sg, xof[i], Xos + (sg_m * (FM * 16) + i * 16) * LDA + kk, LDA);
          }
#pragma unroll
          for (int j = 0; j < FN; j++) {
            joint_matrix_load(sg, pf[j], Ps + (sg_n * (FN * 16) + j * 16) * LDA + kk, LDA);
            joint_matrix_load(sg, gf[j], Gs + (sg_n * (FN * 16) + j * 16) * LDA + kk, LDA);
          }
#pragma unroll
          for (int i = 0; i < FM; i++)
#pragma unroll
            for (int j = 0; j < FN; j++) {
              joint_matrix_mad(sg, accP[i][j], xof[i], pf[j], accP[i][j]);
              joint_matrix_mad(sg, accG[i][j], xif[i], gf[j], accG[i][j]);
            }
        }
        sycl::group_barrier(grp);
      }

      auto Tp = epi.get_multi_ptr<sycl::access::decorated::no>() + sgid * (2 * 16 * LDT);
      auto Tg = Tp + 16 * LDT;
#pragma unroll
      for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) {
          joint_matrix_store(sg, accP[i][j], Tp, LDT, layout::row_major);
          joint_matrix_store(sg, accG[i][j], Tg, LDT, layout::row_major);
          sycl::group_barrier(sg);
          const int mbase = m0 + sg_m * (FM * 16) + i * 16;
          const int cbase = c0 + sg_n * (FN * 16) + j * 16;
          for (int t = lane; t < 256; t += sgw) {
            const int ml = t >> 4, cl = t & 15;
            const int gm = mbase + ml, gc = cbase + cl;
            if (gm >= rows || gc >= D) continue;
            const float proj = Tp[ml * LDT + cl] + p_out_b[gc];
            const float gate = Tg[ml * LDT + cl] + g_out_b[gc];
            out[(size_t)gm * D + gc] = sigmoidf_dev(gate) * proj;
          }
          sycl::group_barrier(sg);
        }
    });
  });
}
#pragma clang diagnostic pop


// ---------------------------------------------------------------------------

struct TriMulBuffers {
  float *x, *mask, *tri, *out;
  bfloat16 *xn, *xout, *a, *b;
  bfloat16 *p_in_w, *g_in_w, *p_out_w, *g_out_w;
  float *norm_in_w, *norm_in_b, *p_in_b, *g_in_b;
  float *norm_out_w, *norm_out_b, *p_out_b, *g_out_b;
};

// One TriMul forward pass: LayerNorm -> gated dual product -> triangle
// projection -> LayerNorm -> gated dual product.
template <int OUTGOING, int SG>
static void trimul_forward(sycl::queue& q, const TriMulBuffers& d,
                           int B, int N, int D, float eps) {
  constexpr int TPB = 256;
  const int rows = B * N * N;

  layernorm_contig_bf16<TPB>(q, rows, d.x, d.xn, d.norm_in_w, d.norm_in_b, D, eps,
                             SG);

  gated_gemm_in_fused<SG>(q, rows, d.xn, d.mask, d.p_in_w, d.p_in_b,
                    d.g_in_w, d.g_in_b, d.a, d.b, B, N, D);

  triangle_batched_bf16<OUTGOING, SG>(q, d.a, d.b, d.tri, N, B * D);

  layernorm_gather_bf16<TPB>(q, rows, d.tri, d.xout, d.norm_out_w, d.norm_out_b,
                             N, D, eps, SG);

  gated_gemm_out_fused<SG>(q, rows, d.xn, d.xout, d.p_out_w, d.p_out_b,
                     d.g_out_w, d.g_out_b, d.out, D);
}

template <int OUTGOING>
static bool trimul_forward(sycl::queue& q, const TriMulBuffers& d,
                           int B, int N, int D, float eps, int sg_size) {
  switch (sg_size) {
    case 8:  trimul_forward<OUTGOING, 8>(q, d, B, N, D, eps);  return true;
    case 16: trimul_forward<OUTGOING, 16>(q, d, B, N, D, eps); return true;
    case 32: trimul_forward<OUTGOING, 32>(q, d, B, N, D, eps); return true;
    case 64: trimul_forward<OUTGOING, 64>(q, d, B, N, D, eps); return true;
    default: return false;
  }
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

  srand(0);
  const size_t xsz = (size_t)B * N * N * D;
  const size_t msz = (size_t)B * N * N;

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
  bfloat16* p_in_w_bf = make_bfloat16(p_in_w, (size_t)2 * D * D);
  bfloat16* g_in_w_bf = make_bfloat16(g_in_w, (size_t)2 * D * D);
  bfloat16* p_out_w_bf = make_bfloat16(p_out_w, (size_t)D * D);
  bfloat16* g_out_w_bf = make_bfloat16(g_out_w, (size_t)D * D);

  float* ref = (float*)xmalloc(xsz * sizeof(float));

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Checked before anything is allocated, so an unsupported device just exits.
  if (!fused::device_supports(q.get_device())) {
    printf("The device does not support the joint_matrix combination "
           "bf16(A) x bf16(B) -> fp32(C/D) with shape 16x16x16\n");
    return 1;
  }

  // A work-group is NSG sub-groups, so the width the device runs sets the
  // work-group size: take the widest the device offers that still fits.
  auto dev = q.get_device();
  const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  const size_t max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
  int sg_size = 0;
  for (int cand : sg_sizes) {
    const bool instantiated =
        cand == 8 || cand == 16 || cand == 32 || cand == 64;
    if (instantiated && fused::NSG * cand <= max_wg && cand > sg_size)
      sg_size = cand;
  }
  if (sg_size == 0) {
    printf("The device reports no sub-group size that fits %d sub-groups into "
           "a work-group\n", fused::NSG);
    return 1;
  }

  bool alloc_ok = true;
  auto DMALLOC = [&](size_t n) {
    float* p = sycl::malloc_device<float>(n, q);
    if (p == nullptr) alloc_ok = false;
    return p;
  };
  auto DMALLOC_BF = [&](size_t n) {
    bfloat16* p = sycl::malloc_device<bfloat16>(n, q);
    if (p == nullptr) alloc_ok = false;
    return p;
  };

  TriMulBuffers d{};
  d.x = DMALLOC(xsz);
  d.mask = DMALLOC(msz);
  d.xn = DMALLOC_BF(xsz);
  d.a = DMALLOC_BF(xsz);
  d.b = DMALLOC_BF(xsz);
  d.tri = DMALLOC(xsz);
  d.xout = DMALLOC_BF(xsz);
  d.out = DMALLOC(xsz);
  d.norm_in_w = DMALLOC(D);
  d.norm_in_b = DMALLOC(D);
  d.p_in_w = DMALLOC_BF((size_t)2 * D * D);
  d.p_in_b = DMALLOC(2 * D);
  d.g_in_w = DMALLOC_BF((size_t)2 * D * D);
  d.g_in_b = DMALLOC(2 * D);
  d.norm_out_w = DMALLOC(D);
  d.norm_out_b = DMALLOC(D);
  d.p_out_w = DMALLOC_BF((size_t)D * D);
  d.p_out_b = DMALLOC(D);
  d.g_out_w = DMALLOC_BF((size_t)D * D);
  d.g_out_b = DMALLOC(D);

  if (!alloc_ok) {
    printf("Failed to allocate the device buffers: each (B,N,N,D) tensor "
           "alone needs %zu bytes\n", xsz * sizeof(float));
    return 1;
  }

  auto H2D = [&](float* dst, const float* src, size_t n) {
    q.memcpy(dst, src, n * sizeof(float));
  };
  auto H2DBF = [&](bfloat16* dst, const bfloat16* src, size_t n) {
    q.memcpy(dst, src, n * sizeof(bfloat16));
  };
  H2D(d.x, x, xsz); H2D(d.mask, mask, msz);
  H2D(d.norm_in_w, norm_in_w, D); H2D(d.norm_in_b, norm_in_b, D);
  H2DBF(d.p_in_w, p_in_w_bf, (size_t)2 * D * D); H2D(d.p_in_b, p_in_b, 2 * D);
  H2DBF(d.g_in_w, g_in_w_bf, (size_t)2 * D * D); H2D(d.g_in_b, g_in_b, 2 * D);
  H2D(d.norm_out_w, norm_out_w, D); H2D(d.norm_out_b, norm_out_b, D);
  H2DBF(d.p_out_w, p_out_w_bf, (size_t)D * D); H2D(d.p_out_b, p_out_b, D);
  H2DBF(d.g_out_w, g_out_w_bf, (size_t)D * D); H2D(d.g_out_b, g_out_b, D);

  int errors = 0;

  // Both directions of the triangle projection are benchmarked in turn.
  for (int outgoing = 1; outgoing >= 0; outgoing--) {
    const char* dir = outgoing ? "outgoing" : "incoming";

    trimul_forward_ref(ref, x, mask, norm_in_w, norm_in_b, p_in_w, p_in_b,
                       g_in_w, g_in_b, norm_out_w, norm_out_b, p_out_w, p_out_b,
                       g_out_w, g_out_b, B, N, D, outgoing, eps);

    // host/device correctness check (run once, verify against the reference)
    // before timing; the same pass also warms up the device
    q.memset(d.out, 0, xsz * sizeof(float));
    try {
      if (outgoing) trimul_forward<1>(q, d, B, N, D, eps, sg_size);
      else          trimul_forward<0>(q, d, B, N, D, eps, sg_size);
      q.wait_and_throw();
    } catch (const sycl::exception& e) {
      printf("Error: %s\n", e.what());
      return 1;
    }
    if (!verify_result(q, d.out, ref, dir, xsz)) errors++;

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      if (outgoing) trimul_forward<1>(q, d, B, N, D, eps, sg_size);
      else          trimul_forward<0>(q, d, B, N, D, eps, sg_size);
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of trimul (%s): %f (us)\n",
           dir, time * 1e-3 / repeat);
  }

  printf("%s\n", errors == 0 ? "PASS" : "FAIL");

  sycl::free(d.x, q); sycl::free(d.mask, q); sycl::free(d.xn, q);
  sycl::free(d.a, q); sycl::free(d.b, q); sycl::free(d.tri, q);
  sycl::free(d.xout, q); sycl::free(d.out, q);
  sycl::free(d.norm_in_w, q); sycl::free(d.norm_in_b, q);
  sycl::free(d.p_in_w, q); sycl::free(d.p_in_b, q);
  sycl::free(d.g_in_w, q); sycl::free(d.g_in_b, q);
  sycl::free(d.norm_out_w, q); sycl::free(d.norm_out_b, q);
  sycl::free(d.p_out_w, q); sycl::free(d.p_out_b, q);
  sycl::free(d.g_out_w, q); sycl::free(d.g_out_b, q);

  free(x); free(mask); free(ref);
  free(norm_in_w); free(norm_in_b); free(p_in_w); free(p_in_b);
  free(g_in_w); free(g_in_b); free(norm_out_w); free(norm_out_b);
  free(p_out_w); free(p_out_b); free(g_out_w); free(g_out_b);
  free(p_in_w_bf); free(g_in_w_bf); free(p_out_w_bf); free(g_out_w_bf);
  return 0;
}
