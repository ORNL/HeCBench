// SYCL TwELL packed-sparse gated MLP for Intel GPUs.
//
// This is the SYCL counterpart of the CUDA/HIP TwELL kernels. The gated FFN
//     gate = relu(x @ GATE^T);  up = x @ UP^T;  out = (gate * up) @ DOWN
// is computed in two TwELL stages:
//
//   D2T (dense -> twell): the gate pre-activation is produced by a bf16 matmul
//   on the Intel XMX matrix engine (oneDNN, with a fused relu post-op), then a
//   SYCL kernel packs the positive entries of every 256-wide feature tile into
//   the TwELL packed format (per row/tile: slot0 = nnz, slot1.. = feature index
//   in the low 16 bits and the bf16 gate value in the high 16 bits).
//
//   gated T2D (twell -> dense): a sub-group sparse kernel (one 32-lane
//   sub-group per token row) walks only the *active* gate features. For each
//   active feature f it computes the up projection <x_row, UP[f]> across the
//   sub-group, scales by the gate value, and scatter-accumulates
//   (gate*up)[f] * DOWN[f] into the dense output row. Work is proportional to
//   the number of nonzero gate activations, not the dense intermediate size.
//
// This mirrors the reference/CUDA semantics: bf16 gate value, fp32-accumulated
// up/down projections.
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace twell {

using bf16 = sycl::ext::oneapi::bfloat16;

static constexpr int T_n = 256;                  // feature tile width
static constexpr int T_n_compressed = T_n / 8;   // 32 packed slots per tile
static constexpr int SG = 32;                    // sub-group width
static constexpr int PER_LANE = T_n / SG;        // 8 elements per lane per chunk

// Packs the dense (already relu'd) gate pre-activation into the TwELL format.
// gate_dense: (M x FEATURE) bf16, row-major. packed: (M x (FEATURE/256)*32) u32.
static sycl::event pack_gate(sycl::queue& q, const bf16* gate_dense,
                             uint32_t* packed, int M, int FEATURE) {
    const int NUM_T_n = FEATURE / T_n;
    return q.parallel_for(
        sycl::range<1>(static_cast<size_t>(M) * NUM_T_n), [=](sycl::id<1> gid) {
            const size_t g0 = gid[0];
            const size_t row = g0 / NUM_T_n;
            const int tile = static_cast<int>(g0 % NUM_T_n);
            const bf16* g = gate_dense + static_cast<long>(row) * FEATURE + tile * T_n;
            uint32_t* out = packed +
                            static_cast<long>(row) * NUM_T_n * T_n_compressed +
                            static_cast<long>(tile) * T_n_compressed;
            uint32_t count = 0;
            for (int c = 0; c < T_n; ++c) {
                const bf16 v = g[c];
                if (static_cast<float>(v) > 0.0f) {
                    const uint16_t bits = sycl::bit_cast<uint16_t>(v);
                    const uint32_t idx =
                        static_cast<uint32_t>(tile * T_n + c) & 0xFFFFu;
                    // wrap within this tile's payload slots [1, 31]
                    const uint32_t slot = (count % (T_n_compressed - 1)) + 1;
                    out[slot] = idx | (static_cast<uint32_t>(bits) << 16);
                    ++count;
                }
            }
            out[0] = count;
        });
}

// bf16-rounded product of two bf16 values (mirrors CUDA __hmul: the product is
// rounded to bf16). Used by the default (non high-precision) packed path.
static inline bf16 bf16_mul(bf16 a, bf16 b) {
    return static_cast<bf16>(static_cast<float>(a) * static_cast<float>(b));
}

// Gated packed-to-dense sparse projection.
//   IN     : (M x OUT_DIM) bf16 dense token activations (row-major)
//   packed : (M x (FEATURE/256)*32) u32 TwELL gate
//   UP     : (FEATURE x OUT_DIM) bf16  (up projection weights)
//   DOWN   : (FEATURE x OUT_DIM) bf16  (down projection weights)
//   OUT    : (M x OUT_DIM) bf16 dense output (row-major)
// One 32-lane sub-group computes one row; each lane owns OUT_DIM/32 columns.
//
// HIGH_PRECISION selects the arithmetic, mirroring the two CUDA TwELL kernels:
//   * false (default): bf16 products with fp32 accumulation, and the gate*up
//     feature is rounded to bf16 before the DOWN projection (matches
//     TWELL_GATED_T2D::mm_t2d_kernel).
//   * true: fp32 products/accumulation throughout, keeping the gate*up feature
//     in fp32 (matches mm_t2d_high_precision_kernel).
template <bool HIGH_PRECISION>
static sycl::event gated_t2d_impl(sycl::queue& q, const bf16* IN,
                                  const uint32_t* packed, const bf16* UP,
                                  const bf16* DOWN, bf16* OUT, int M,
                                  int FEATURE, int OUT_DIM) {
    const int NUM_T_n = FEATURE / T_n;
    const int ITERS = OUT_DIM / T_n;  // 256-wide column chunks per row
    sycl::nd_range<1> ndr(static_cast<size_t>(M) * SG, SG);
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(
            ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                auto sg = it.get_sub_group();
                const int lane = static_cast<int>(it.get_local_id(0));
                const size_t row = it.get_group(0);

                const bf16* in_row = IN + static_cast<long>(row) * OUT_DIM;
                const uint32_t* prow =
                    packed + static_cast<long>(row) * NUM_T_n * T_n_compressed;
                bf16* out_row = OUT + static_cast<long>(row) * OUT_DIM;

                // cache this lane's slice of the dense input row in registers
                bf16 in_cache[16][PER_LANE];  // ITERS <= 16 for OUT_DIM<=4096
                for (int i = 0; i < ITERS; ++i)
                    for (int e = 0; e < PER_LANE; ++e)
                        in_cache[i][e] = in_row[i * T_n + lane * PER_LANE + e];

                float out_acc[16][PER_LANE];
                for (int i = 0; i < ITERS; ++i)
                    for (int e = 0; e < PER_LANE; ++e) out_acc[i][e] = 0.0f;

                for (int tile = 0; tile < NUM_T_n; ++tile) {
                    const int lane_reg =
                        static_cast<int>(prow[tile * T_n_compressed + lane]);
                    int nnz = sycl::select_from_group(sg, lane_reg, 0);
                    if (nnz > T_n_compressed - 1) nnz = T_n_compressed - 1;

                    for (int idx = 1; idx <= nnz; ++idx) {
                        const uint32_t comp =
                            static_cast<uint32_t>(sycl::select_from_group(sg, lane_reg, idx));
                        const uint32_t f = comp & 0xFFFFu;  // feature index
                        const bf16 gate_val =
                            sycl::bit_cast<bf16>(static_cast<uint16_t>(comp >> 16));

                        // up projection: <in_row, UP[f]> over OUT_DIM columns
                        const bf16* up_f = UP + static_cast<long>(f) * OUT_DIM;
                        float up_acc = 0.0f;
                        for (int i = 0; i < ITERS; ++i)
                            for (int e = 0; e < PER_LANE; ++e) {
                                const bf16 w = up_f[i * T_n + lane * PER_LANE + e];
                                if constexpr (HIGH_PRECISION)
                                    up_acc += static_cast<float>(in_cache[i][e]) *
                                              static_cast<float>(w);
                                else
                                    up_acc += static_cast<float>(
                                        bf16_mul(in_cache[i][e], w));
                            }
                        // reduce the partials across the sub-group
                        for (int s = SG / 2; s > 0; s >>= 1)
                            up_acc += sycl::permute_group_by_xor(sg, up_acc, s);

                        const bf16* down_f = DOWN + static_cast<long>(f) * OUT_DIM;
                        if constexpr (HIGH_PRECISION) {
                            // gate * up kept in fp32; down product in fp32
                            const float feat = static_cast<float>(gate_val) * up_acc;
                            for (int i = 0; i < ITERS; ++i)
                                for (int e = 0; e < PER_LANE; ++e)
                                    out_acc[i][e] +=
                                        feat * static_cast<float>(
                                                   down_f[i * T_n + lane * PER_LANE + e]);
                        } else {
                            // gate * up rounded to bf16; down product in bf16
                            const bf16 feat =
                                bf16_mul(gate_val, static_cast<bf16>(up_acc));
                            for (int i = 0; i < ITERS; ++i)
                                for (int e = 0; e < PER_LANE; ++e)
                                    out_acc[i][e] += static_cast<float>(bf16_mul(
                                        feat, down_f[i * T_n + lane * PER_LANE + e]));
                        }
                    }
                }

                for (int i = 0; i < ITERS; ++i)
                    for (int e = 0; e < PER_LANE; ++e)
                        out_row[i * T_n + lane * PER_LANE + e] =
                            static_cast<bf16>(out_acc[i][e]);
            });
    });
}

// default packed path: bf16 products (matches CUDA mm_t2d_kernel)
static sycl::event gated_t2d(sycl::queue& q, const bf16* IN,
                             const uint32_t* packed, const bf16* UP,
                             const bf16* DOWN, bf16* OUT, int M, int FEATURE,
                             int OUT_DIM) {
    return gated_t2d_impl<false>(q, IN, packed, UP, DOWN, OUT, M, FEATURE,
                                 OUT_DIM);
}

// high-precision packed path: fp32 products (matches CUDA
// mm_t2d_high_precision_kernel)
static sycl::event gated_t2d_high_precision(sycl::queue& q, const bf16* IN,
                                            const uint32_t* packed,
                                            const bf16* UP, const bf16* DOWN,
                                            bf16* OUT, int M, int FEATURE,
                                            int OUT_DIM) {
    return gated_t2d_impl<true>(q, IN, packed, UP, DOWN, OUT, M, FEATURE,
                                OUT_DIM);
}

}  // namespace twell
