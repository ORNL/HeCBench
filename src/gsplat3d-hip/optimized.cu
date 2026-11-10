// Derived from the 3DGS rasterization kernels in gsplat
// (https://github.com/nerfstudio-project/gsplat), Apache-2.0, see LICENSE.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include "launch.h"
#include "raster_math.h"
#include "warp_sum.cuh"

namespace cg = cooperative_groups;

// Return the float held by the lane whose id differs from this one by MASK,
// inside the same group of kWaveSize lanes. MASK 1, 2, 4, and 8 stay inside a
// 16-lane row. MASK 16 exchanges the two 16-lane halves of that group. The
// bits travel as an integer, so the value itself is unchanged.
//
// gfx10 and later do the row moves with DPP row_xmask and the half exchange
// with permlanex16. Those targets have 32-lane waves, so the group is the
// whole wave. gfx9, including gfx90a and gfx942, has neither instruction and
// uses ds_bpermute. The lane index is masked to kWaveSize, so on a 64-wide
// wave the exchange stays inside lanes 0-31 or lanes 32-63.
template <int MASK>
__device__ __forceinline__ float wave32_xor(float v) {
    const int iv = __float_as_int(v);
    int r;
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__))
    if constexpr (MASK == 16) {
        r = __builtin_amdgcn_permlanex16(iv, iv, 0x76543210u, 0xfedcba98u, false, false);
    } else {
        r = __builtin_amdgcn_update_dpp(iv, iv, 0x160 | MASK, 0xf, 0xf, false);
    }
#else
    const int self = __lane_id();
    const int index = (self & ~(kWaveSize - 1)) | ((self ^ MASK) & (kWaveSize - 1));
    r = __builtin_amdgcn_ds_bpermute(index << 2, iv);
#endif
    return __int_as_float(r);
}

// One butterfly step of the transposed wave reduction. `in` holds 2H channel
// values. This lane keeps H of them and adds the other H taken from the lane
// MASK away. `upper` chooses which half is sent.
// Both halves are read before the select. A select between two array
// elements lets ROCm clang 19 and 20 index the array dynamically, which
// expands into a 16-way compare chain and doubles the kernel's run time.
template <int MASK, int H>
__device__ __forceinline__ void wave32_transpose_round(const float (&in)[2 * H], float (&out)[H], bool upper) {
#pragma unroll
    for (int k = 0; k < H; ++k) {
        const float lo = in[k];
        const float hi = in[k + H];
        const float send = upper ? lo : hi;
        const float keep = upper ? hi : lo;
        out[k] = keep + wave32_xor<MASK>(send);
    }
}

// Sum 16 channels across one 32-lane group. Lane k and lane k+16 both receive
// the sum of channel k. Channels 9..15 are padding and remain zero. On a
// 64-wide wave the upper 32 lanes are a second group and reduce separately.
__device__ __forceinline__ float wave32_transpose_reduce16(const float (&a)[16], int lane) {
    float b[8], c[4], d[2], e[1];
    wave32_transpose_round<8, 8>(a, b, lane & 8);
    wave32_transpose_round<4, 4>(b, c, lane & 4);
    wave32_transpose_round<2, 2>(c, d, lane & 2);
    wave32_transpose_round<1, 1>(d, e, lane & 1);
    return e[0] + wave32_xor<16>(e[0]);
}

// Lane k < 9 atomicAdds its reduced channel into shared-memory slot t.
// The other lanes hold padding or the copy of the sum from the other 16-lane
// half of this group, and do not write.
__device__ __forceinline__ void sink_gaussian_grads9_lds(float* acc, int t, float val, int lane) {
    if (lane < 9) {
        __hip_atomic_fetch_add(acc + t * 9 + lane, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    }
}

// Shared-memory staging for one batch of per-Gaussian gradients.
// Each slot holds CDIM color channels plus 3 conic, 2 mean, and 1 opacity.
template <int CDIM>
struct GradLdsSink {
    static constexpr int NCOMP = CDIM + 6;

    // AtomicAdd every component into slot t. Workgroup scope, relaxed order.
    // Used by the rocprim path, where only lane 0 holds the group sum.
    static __device__ __forceinline__ void add(float* acc, int t, const float (&v_rgb)[CDIM],
                                               const vec3& v_conic, const vec2& v_xy, float v_opacity) {
        float* slot = acc + t * NCOMP;
#pragma unroll
        for (int k = 0; k < CDIM; ++k) {
            __hip_atomic_fetch_add(slot + k, v_rgb[k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        }
        __hip_atomic_fetch_add(slot + CDIM + 0, v_conic.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        __hip_atomic_fetch_add(slot + CDIM + 1, v_conic.y, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        __hip_atomic_fetch_add(slot + CDIM + 2, v_conic.z, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        __hip_atomic_fetch_add(slot + CDIM + 3, v_xy.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        __hip_atomic_fetch_add(slot + CDIM + 4, v_xy.y, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        __hip_atomic_fetch_add(slot + CDIM + 5, v_opacity, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    }

    // Move slot `slot_idx` into the global gradient of Gaussian g and clear it.
    // Zero components are skipped. The global atomicAdd combines this tile
    // with every other tile that touched g.
    static __device__ __forceinline__ void flush(float* acc, int slot_idx, int g, float* v_colors,
                                                 float* v_conics, float* v_means2d, float* v_opacities) {
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
// 32-lane group in one pass; otherwise each channel uses rocprim.
// 1/(1 - alpha) is a hardware reciprocal. Each group adds into a
// shared-memory slot, and the block flushes that batch to global memory
// once per Gaussian.
template <int CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_bwd_optimized(
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
    auto block = cg::this_thread_block();
    int tile_id = block.group_index().y * tile_width + block.group_index().z;
    int i = block.group_index().y * tile_size + block.thread_index().y;
    int j = block.group_index().z * tile_size + block.thread_index().x;

    const float px = j + 0.5f;
    const float py = i + 0.5f;
    const int pix_id = min(i * image_width + j, image_width * image_height - 1);
    bool inside = (i < image_height && j < image_width);

    int range_start = tile_offsets[tile_id];
    int range_end = (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int block_size = block.size();
    const int batch_allocation_size = max_batch_size;

    extern __shared__ int s[];
    int* id_batch = s;
    vec3* xy_opacity_batch = reinterpret_cast<vec3*>(&id_batch[batch_allocation_size]);
    vec3* conic_batch = reinterpret_cast<vec3*>(&xy_opacity_batch[batch_allocation_size]);
    float* rgbs_batch = (float*)&conic_batch[batch_allocation_size];

    using warp_reduce_float_t = rocprim::warp_reduce<float, kWaveSize>;
    auto* warp_storage_base =
        (typename warp_reduce_float_t::storage_type*)(rgbs_batch + batch_allocation_size * CDIM);
    const int warps_per_block = (block_size + kWaveSize - 1) / kWaveSize;
    // s is 16-byte aligned, so rounding the byte offset aligns grad_acc. The
    // offset is taken from s rather than from the address so the compiler
    // still sees grad_acc as shared memory.
    const size_t acc_offset =
        (reinterpret_cast<char*>(warp_storage_base + warps_per_block) - reinterpret_cast<char*>(s) + 15) &
        ~size_t(15);
    float* grad_acc = reinterpret_cast<float*>(reinterpret_cast<char*>(s) + acc_offset);
    for (int e = block.thread_rank(); e < batch_allocation_size * GradLdsSink<CDIM>::NCOMP;
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

    const int tr = block.thread_rank();
    cg::thread_block_tile<kWaveSize> warp = cg::tiled_partition<kWaveSize>(block);

    // The third template argument broadcasts the max to every lane in the
    // 32-lane group. The default leaves the full max on lane 0, and the other
    // lanes then stop the backward walk too early.
    using warp_reduce_bin_t = rocprim::warp_reduce<int, kWaveSize, true>;
    __shared__ typename warp_reduce_bin_t::storage_type warp_storage[kTileSize * kTileSize / kWaveSize];
    warp_reduce_bin_t wreduce;
    int warp_bin_final;
    wreduce.reduce(bin_final, warp_bin_final, warp_storage[tr / kWaveSize], rocprim::maximum<int>());

    __shared__ int s_wave_bin_final[32];
    const int num_waves = (block_size + kWaveSize - 1) / kWaveSize;
    if (warp.thread_rank() == 0) {
        s_wave_bin_final[tr / kWaveSize] = warp_bin_final;
    }
    block.sync();
    int block_bin_final = s_wave_bin_final[0];
    for (int w = 1; w < num_waves; ++w) {
        block_bin_final = max(block_bin_final, s_wave_bin_final[w]);
    }
    range_end = min(range_end, block_bin_final + 1);
    const int num_batches = range_end > range_start
                                     ? (range_end - range_start + max_batch_size - 1) / max_batch_size
                                     : 0;

    for (int b = 0; b < num_batches; ++b) {
        block.sync();

        if (b > 0 && tr < batch_allocation_size) {
            GradLdsSink<CDIM>::flush(grad_acc, tr, id_batch[tr], v_colors, (float*)v_conics,
                                     (float*)v_means2d, v_opacities);
        }
        const int batch_end = range_end - 1 - max_batch_size * b;
        const int current_batch_size = min(max_batch_size, batch_end + 1 - range_start);
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
        block.sync();

        for (int t = max(0, batch_end - warp_bin_final); t < current_batch_size; ++t) {
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

            if (!logical_warp_any(valid)) {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            if (valid) {
                float ra = __builtin_amdgcn_rcpf(1.0f - alpha);
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
                // Lane within this 32-lane group. On a 64-wide wave __lane_id()
                // runs to 63, and lanes 32-63 are a second group whose sums
                // also have to be published.
                const int lane = __lane_id() & (kWaveSize - 1);
                const float vals[16] = {v_rgb_local[0], v_rgb_local[1], v_rgb_local[2], v_conic_local.x,
                                        v_conic_local.y, v_conic_local.z, v_xy_local.x,    v_xy_local.y,
                                        v_opacity_local, 0.f,             0.f,             0.f,
                                        0.f,             0.f,             0.f,             0.f};
                const float red = wave32_transpose_reduce16(vals, lane);
                sink_gaussian_grads9_lds(grad_acc, t, red, lane);
                continue;
            }
            warpSum<CDIM, kWaveSize>(v_rgb_local, warp_storage_base);
            warpSum<kWaveSize>(v_conic_local, warp_storage_base);
            warpSum<kWaveSize>(v_xy_local, warp_storage_base);
            warpSum<kWaveSize>(v_opacity_local, warp_storage_base);
            if (warp.thread_rank() == 0) {
                GradLdsSink<CDIM>::add(grad_acc, t, v_rgb_local, v_conic_local, v_xy_local, v_opacity_local);
            }
        }
    }
    if (num_batches > 0) {
        block.sync();
        const int last_batch_size = range_end - range_start - max_batch_size * (num_batches - 1);
        if (tr < last_batch_size) {
            GradLdsSink<CDIM>::flush(grad_acc, tr, id_batch[tr], v_colors, (float*)v_conics,
                                     (float*)v_means2d, v_opacities);
        }
    }
}

// Same batch size as the baseline. Shared memory also holds the
// nine-component accumulator, with 16 bytes of padding so it is aligned.
static int optimized_batch_size(size_t* shmem) {
    const int block_size = kTileSize * kTileSize;
    const int max_batch_size = block_size;
    const int warps_per_block = (block_size + kWaveSize - 1) / kWaveSize;
    const size_t warp_scratch_bytes =
        warps_per_block * sizeof(rocprim::warp_reduce<float, kWaveSize>::storage_type);
    *shmem = max_batch_size * (sizeof(int) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * kCdim) +
             warp_scratch_bytes;
    *shmem = *shmem + 16 + max_batch_size * sizeof(float) * GradLdsSink<kCdim>::NCOMP;
    return max_batch_size;
}

// Launch the optimized backward kernel. Does nothing when the scene has no
// intersections.
void launch_rasterize_optimized(const BwdArgs& args) {
    if (args.n_isects == 0) {
        return;
    }
    size_t shmem = 0;
    const int max_batch_size = optimized_batch_size(&shmem);
    auto kernel = rasterize_to_pixels_3dgs_bwd_optimized<kCdim, float>;
    const dim3 threads(kTileSize, kTileSize, 1);
    const dim3 grid(1, args.tile_height, args.tile_width);
    kernel<<<grid, threads, shmem>>>(
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
}
