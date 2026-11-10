// Derived from the 3DGS rasterization kernels in gsplat
// (https://github.com/nerfstudio-project/gsplat), Apache-2.0, see LICENSE.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <sycl/sycl.hpp>

#include "launch.h"
#include "raster_math.h"
#include "warp_sum.h"

// Return the float held by the lane whose id differs from this one by MASK,
// inside the same group of kWaveSize lanes. MASK 1, 2, 4, and 8 stay inside a
// 16-lane row. MASK 16 exchanges the two 16-lane halves of that group. The
// NVIDIA path is shfl.sync.bfly, the same instruction as CUDA __shfl_xor_sync.
// The AMD path is the HIP DPP / ds_bpermute sequence. MASK is below kWaveSize,
// so on a 64-wide sub-group the partner stays inside lanes 0-31 or 32-63.
template <int MASK>
inline float wave32_xor(const sycl::sub_group& sg, float v) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    (void)sg;
    float r;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, 0xffffffff;" : "=f"(r) : "f"(v), "r"(MASK));
    return r;
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
    const int iv = sycl::bit_cast<int>(v);
    int r;
#if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
    (void)sg;
    if constexpr (MASK == 16) {
        r = __builtin_amdgcn_permlanex16(iv, iv, 0x76543210u, 0xfedcba98u, false, false);
    } else {
        r = __builtin_amdgcn_update_dpp(iv, iv, 0x160 | MASK, 0xf, 0xf, false);
    }
#else
    const int self = static_cast<int>(sg.get_local_linear_id());
    const int index = (self & ~(kWaveSize - 1)) | ((self ^ MASK) & (kWaveSize - 1));
    r = __builtin_amdgcn_ds_bpermute(index << 2, iv);
#endif
    return sycl::bit_cast<float>(r);
#else
    return sycl::permute_group_by_xor(sg, v, MASK);
#endif
}

// One butterfly step of the transposed group reduction. `in` holds 2H channel
// values. This lane keeps H of them and adds the other H taken from the lane
// MASK away. `upper` chooses which half is sent.
// Both halves are read before the select. A select between two array
// elements lets ROCm clang 19 and 20 index the array dynamically, which
// expands into a 16-way compare chain and doubles the kernel's run time.
template <int MASK, int H>
inline void wave32_transpose_round(const sycl::sub_group& sg, const float (&in)[2 * H], float (&out)[H],
                                   bool upper) {
#pragma unroll
    for (int k = 0; k < H; ++k) {
        const float lo = in[k];
        const float hi = in[k + H];
        const float send = upper ? lo : hi;
        const float keep = upper ? hi : lo;
        out[k] = keep + wave32_xor<MASK>(sg, send);
    }
}

// Sum 16 channels across one 32-lane group. Lane k and lane k+16 both receive
// the sum of channel k. Channels 9..15 are padding and remain zero. On a
// 64-wide sub-group the upper 32 lanes are a second group and reduce
// separately.
inline float wave32_transpose_reduce16(const sycl::sub_group& sg, const float (&a)[16], int lane) {
    float b[8], c[4], d[2], e[1];
    wave32_transpose_round<8, 8>(sg, a, b, lane & 8);
    wave32_transpose_round<4, 4>(sg, b, c, lane & 4);
    wave32_transpose_round<2, 2>(sg, c, d, lane & 2);
    wave32_transpose_round<1, 1>(sg, d, e, lane & 1);
    return e[0] + wave32_xor<16>(sg, e[0]);
}

// Lane k < 9 atomicAdds its reduced channel into local-memory slot t.
// The other lanes hold padding or the copy of the sum from the other 16-lane
// half of this group, and do not write.
inline void sink_gaussian_grads9_lds(float* acc, int t, float val, int lane) {
    if (lane < 9) {
        atomicAdd_block(acc + t * 9 + lane, val);
    }
}

// Local-memory staging for one batch of per-Gaussian gradients.
// Each slot holds CDIM color channels plus 3 conic, 2 mean, and 1 opacity.
template <int CDIM>
struct GradLdsSink {
    static constexpr int NCOMP = CDIM + 6;

    // AtomicAdd every component into slot t. Work-group scope, relaxed order.
    // Used by the warpSum path.
    static inline void add(float* acc, int t, const float (&v_rgb)[CDIM], const vec3& v_conic,
                           const vec2& v_xy, float v_opacity) {
        float* slot = acc + t * NCOMP;
#pragma unroll
        for (int k = 0; k < CDIM; ++k) {
            atomicAdd_block(slot + k, v_rgb[k]);
        }
        atomicAdd_block(slot + CDIM + 0, v_conic.x);
        atomicAdd_block(slot + CDIM + 1, v_conic.y);
        atomicAdd_block(slot + CDIM + 2, v_conic.z);
        atomicAdd_block(slot + CDIM + 3, v_xy.x);
        atomicAdd_block(slot + CDIM + 4, v_xy.y);
        atomicAdd_block(slot + CDIM + 5, v_opacity);
    }

    // Move slot `slot_idx` into the global gradient of Gaussian g and clear it.
    // Zero components are skipped. The global atomicAdd combines this tile
    // with every other tile that touched g.
    static inline void flush(float* acc, int slot_idx, int g, float* v_colors, float* v_conics,
                             float* v_means2d, float* v_opacities) {
        float* slot = acc + slot_idx * NCOMP;
        float v[NCOMP];
#pragma unroll
        for (int k = 0; k < NCOMP; ++k) {
            v[k] = slot[k];
            slot[k] = 0.f;
        }
#pragma unroll
        for (int k = 0; k < CDIM; ++k) {
            if (v[k] != 0.f) atomicAdd(v_colors + CDIM * g + k, v[k]);
        }
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            if (v[CDIM + k] != 0.f) atomicAdd(v_conics + 3 * g + k, v[CDIM + k]);
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            if (v[CDIM + 3 + k] != 0.f) atomicAdd(v_means2d + 2 * g + k, v[CDIM + 3 + k]);
        }
        if (v[CDIM + 5] != 0.f) atomicAdd(v_opacities + g, v[CDIM + 5]);
    }
};

// Backward kernel computing the same gradients as the baseline.
// The tile walk stops at the farthest Gaussian any pixel in the tile still
// needs. For three color channels the nine values are summed across the
// 32-lane group in one pass; otherwise each channel uses warpSum.
// 1/(1 - alpha) is a native reciprocal. Each group adds into a local-memory
// slot, and the work-group flushes that batch to global memory once per
// Gaussian.
template <int CDIM, typename scalar_t>
void rasterize_to_pixels_3dgs_bwd_optimized(
    const sycl::nd_item<3>& item,
    int* s,
    int* s_wave_bin_final,
    const int n_isects,
    const vec2* __restrict__ means2d,
    const vec3* __restrict__ conics,
    const scalar_t* __restrict__ colors,
    const scalar_t* __restrict__ opacities,
    const int image_width,
    const int image_height,
    const int tile_size,
    const int tile_width,
    const int tile_height,
    const int* __restrict__ tile_offsets,
    const int* __restrict__ flatten_ids,
    const scalar_t* __restrict__ render_alphas,
    const int* __restrict__ last_ids,
    const scalar_t* __restrict__ v_render_colors,
    const scalar_t* __restrict__ v_render_alphas,
    vec2* __restrict__ v_means2d,
    vec3* __restrict__ v_conics,
    scalar_t* __restrict__ v_colors,
    scalar_t* __restrict__ v_opacities,
    const int max_batch_size) {
    auto block = item.get_group();
    const sycl::sub_group sg = item.get_sub_group();
    int tile_id = item.get_group(1) * tile_width + item.get_group(2);
    int i = item.get_group(1) * tile_size + item.get_local_id(1);
    int j = item.get_group(2) * tile_size + item.get_local_id(2);

    const float px = j + 0.5f;
    const float py = i + 0.5f;
    const int pix_id = sycl::min(i * image_width + j, image_width * image_height - 1);
    bool inside = (i < image_height && j < image_width);

    int range_start = tile_offsets[tile_id];
    int range_end = (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int block_size = item.get_local_range().size();
    const int batch_allocation_size = max_batch_size;

    int* id_batch = s;
    vec3* xy_opacity_batch = reinterpret_cast<vec3*>(&id_batch[batch_allocation_size]);
    vec3* conic_batch = reinterpret_cast<vec3*>(&xy_opacity_batch[batch_allocation_size]);
    float* rgbs_batch = (float*)&conic_batch[batch_allocation_size];

    auto* warp_storage_base = (warp_reduce_storage*)(rgbs_batch + batch_allocation_size * CDIM);
    const int warps_per_block = (block_size + kWaveSize - 1) / kWaveSize;
    // s is 16-byte aligned, so rounding the byte offset aligns grad_acc. The
    // offset is taken from s rather than from the address so the compiler
    // still sees grad_acc as local memory.
    const size_t acc_offset =
        (reinterpret_cast<char*>(warp_storage_base + warps_per_block) - reinterpret_cast<char*>(s) + 15) &
        ~size_t(15);
    float* grad_acc = reinterpret_cast<float*>(reinterpret_cast<char*>(s) + acc_offset);
    for (int e = item.get_local_linear_id(); e < batch_allocation_size * GradLdsSink<CDIM>::NCOMP;
         e += block_size) {
        grad_acc[e] = 0.f;
    }

    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    float buffer[CDIM] = {0.f};
    const int bin_final = inside ? last_ids[pix_id] : 0;

    float v_render_c[CDIM];
#pragma unroll
    for (int k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    const int tr = item.get_local_linear_id();

    // Every lane in the 32-lane group receives the max. If only lane 0 had
    // it, the other lanes would stop the backward walk too early.
    int warp_bin_final = logical_warp_max(sg, bin_final);

    const int num_waves = (block_size + kWaveSize - 1) / kWaveSize;
    if (logical_warp_lane(sg) == 0) {
        s_wave_bin_final[logical_warp_id(sg)] = warp_bin_final;
    }
    sycl::group_barrier(block);
    int block_bin_final = s_wave_bin_final[0];
    for (int w = 1; w < num_waves; ++w) {
        block_bin_final = sycl::max(block_bin_final, s_wave_bin_final[w]);
    }
    range_end = sycl::min(range_end, block_bin_final + 1);
    const int num_batches = range_end > range_start
                                     ? (range_end - range_start + max_batch_size - 1) / max_batch_size
                                     : 0;

    for (int b = 0; b < num_batches; ++b) {
        sycl::group_barrier(block);

        if (b > 0 && tr < batch_allocation_size) {
            GradLdsSink<CDIM>::flush(grad_acc, tr, id_batch[tr], v_colors, (float*)v_conics,
                                     (float*)v_means2d, v_opacities);
        }
        const int batch_end = range_end - 1 - max_batch_size * b;
        const int current_batch_size = sycl::min(max_batch_size, batch_end + 1 - range_start);
        const int idx = batch_end - tr;
        if (tr < current_batch_size && idx >= range_start) {
            int g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (int k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        sycl::group_barrier(block);

        for (int t = sycl::max(0, batch_end - warp_bin_final); t < current_batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float opac_vis = 0.f;
            vec2 delta;
            vec3 conic;
            float vis;

            if (valid) {
                conic = conic_batch[t];
                vec3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                const GsplatEval eval =
                    gsplat_eval(conic.x, conic.y, conic.z, delta.x, delta.y, opac);
                vis = eval.vis;
                opac_vis = eval.opac_vis;
                alpha = eval.alpha;
                if (eval.sigma < 0.f || alpha < kAlphaThreshold) {
                    valid = false;
                }
            }

            if (!logical_warp_any(sg, valid)) {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            if (valid) {
                float ra = sycl::native::recip(1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;
#pragma unroll
                for (int k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                float v_alpha = 0.f;
#pragma unroll
                for (int k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) * v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;

                if (opac_vis <= 0.999f) {
                    const float v_sigma = -opac_vis * v_alpha;
                    v_conic_local = {0.5f * v_sigma * delta.x * delta.x, v_sigma * delta.x * delta.y,
                                     0.5f * v_sigma * delta.y * delta.y};
                    v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                  v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                    v_opacity_local = vis * v_alpha;
                }

#pragma unroll
                for (int k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
            if constexpr (CDIM == 3) {
                // Lane within this 32-lane group. On a 64-wide sub-group the
                // sub-group lane id runs to 63, and lanes 32-63 are a second
                // group whose sums also have to be published.
                const int lane = logical_warp_lane(sg);
                const float vals[16] = {v_rgb_local[0], v_rgb_local[1], v_rgb_local[2], v_conic_local.x,
                                        v_conic_local.y, v_conic_local.z, v_xy_local.x,    v_xy_local.y,
                                        v_opacity_local, 0.f,             0.f,             0.f,
                                        0.f,             0.f,             0.f,             0.f};
                const float red = wave32_transpose_reduce16(sg, vals, lane);
                sink_gaussian_grads9_lds(grad_acc, t, red, lane);
                continue;
            }
            warpSum<CDIM, kWaveSize>(sg, v_rgb_local, warp_storage_base);
            warpSum<kWaveSize>(sg, v_conic_local, warp_storage_base);
            warpSum<kWaveSize>(sg, v_xy_local, warp_storage_base);
            warpSum<kWaveSize>(sg, v_opacity_local, warp_storage_base);
            if (logical_warp_lane(sg) == 0) {
                GradLdsSink<CDIM>::add(grad_acc, t, v_rgb_local, v_conic_local, v_xy_local, v_opacity_local);
            }
        }
    }
    if (num_batches > 0) {
        sycl::group_barrier(block);
        const int last_batch_size = range_end - range_start - max_batch_size * (num_batches - 1);
        if (tr < last_batch_size) {
            GradLdsSink<CDIM>::flush(grad_acc, tr, id_batch[tr], v_colors, (float*)v_conics,
                                     (float*)v_means2d, v_opacities);
        }
    }
}

// Same batch size as the baseline. Local memory also holds the
// nine-component accumulator, with 16 bytes of padding so it is aligned.
static int optimized_batch_size(size_t* shmem) {
    const int block_size = kTileSize * kTileSize;
    const int max_batch_size = block_size;
    const int warps_per_block = (block_size + kWaveSize - 1) / kWaveSize;
    const size_t warp_scratch_bytes = warps_per_block * sizeof(warp_reduce_storage);
    *shmem = max_batch_size * (sizeof(int) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * kCdim) +
             warp_scratch_bytes;
    *shmem = *shmem + 16 + max_batch_size * sizeof(float) * GradLdsSink<kCdim>::NCOMP;
    return max_batch_size;
}

template <int SG>
static void launch_rasterize_optimized_sg(sycl::queue& q, const BwdArgs& args, const int max_batch_size,
                                          const size_t shmem) {
    const sycl::range<3> threads(1, kTileSize, kTileSize);
    const sycl::range<3> grid(1, args.tile_height * kTileSize, args.tile_width * kTileSize);
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::int4, 1> smem(sycl::range<1>((shmem + 15) / 16), cgh);
        sycl::local_accessor<int, 1> s_wave_bin_final(sycl::range<1>(32), cgh);
        cgh.parallel_for(sycl::nd_range<3>(grid, threads), [=](sycl::nd_item<3> item)
                                                               [[sycl::reqd_sub_group_size(SG)]] {
            rasterize_to_pixels_3dgs_bwd_optimized<kCdim, float>(
                item,
                reinterpret_cast<int*>(smem.get_multi_ptr<sycl::access::decorated::no>().get()),
                s_wave_bin_final.get_multi_ptr<sycl::access::decorated::no>().get(),
                args.n_isects,
                args.means2d,
                args.conics,
                args.colors,
                args.opacities,
                args.image_width,
                args.image_height,
                kTileSize,
                args.tile_width,
                args.tile_height,
                args.tile_offsets,
                args.flatten_ids,
                args.render_alphas,
                args.last_ids,
                args.v_render_colors,
                args.v_render_alphas,
                args.v_means2d,
                args.v_conics,
                args.v_colors,
                args.v_opacities,
                max_batch_size);
        });
    });
}

// Launch the optimized backward kernel. Does nothing when the scene has no
// intersections. Aborts if the local memory optimized_batch_size computed,
// plus s_wave_bin_final, exceeds the device limit.
void launch_rasterize_optimized(sycl::queue& q, const BwdArgs& args) {
    if (args.n_isects == 0) {
        return;
    }
    size_t shmem = 0;
    const int max_batch_size = optimized_batch_size(&shmem);
    const size_t local_mem = q.get_device().get_info<sycl::info::device::local_mem_size>();
    if (shmem + 32 * sizeof(int) > local_mem) {
        fprintf(stderr, "SYCL error %s:%d local memory: need %zu bytes, device has %zu\n", __FILE__, __LINE__,
                shmem + 32 * sizeof(int), local_mem);
        exit(EXIT_FAILURE);
    }
    if (args.sub_group_size == 32) {
        launch_rasterize_optimized_sg<32>(q, args, max_batch_size, shmem);
    } else {
        launch_rasterize_optimized_sg<64>(q, args, max_batch_size, shmem);
    }
}
