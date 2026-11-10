// Derived from the 3DGS rasterization kernels in gsplat
// (https://github.com/nerfstudio-project/gsplat), Apache-2.0, see LICENSE.
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include "launch.h"
#include "raster_math.h"
#include "warp_sum.cuh"

namespace cg = cooperative_groups;

// gsplat backward kernel.
// One 16x16 block per tile walks that tile's Gaussians from far to near.
// Lanes cooperate in groups of kWaveSize. On a 32-wide wave that group is
// the whole wave. On a 64-wide wave the two halves are separate groups.
// Transmittance starts at 1 - render_alpha and is rebuilt by multiplying by
// 1/(1 - alpha) at each blended Gaussian. The farthest intersection any lane
// in the group still needs is broadcast to every lane in that group, so the
// walk stays convergent. Each group sums the nine gradient channels with
// rocprim, and lane 0 atomicAdds that sum into the global gradients. Mean,
// conic, and opacity gradients are zero where opacity * exp(-sigma) is above
// the 0.999 clamp.
template <int CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_bwd_baseline(
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
    const int batch_allocation_size = max_batch_size;
    const int num_batches =
        (range_end - range_start + max_batch_size - 1) / max_batch_size;

    extern __shared__ int s[];
    int* id_batch = s;
    vec3* xy_opacity_batch = reinterpret_cast<vec3*>(&id_batch[batch_allocation_size]);
    vec3* conic_batch = reinterpret_cast<vec3*>(&xy_opacity_batch[batch_allocation_size]);
    float* rgbs_batch = (float*)&conic_batch[batch_allocation_size];

    using warp_reduce_float_t = rocprim::warp_reduce<float, kWaveSize>;
    auto* warp_storage_base = (typename warp_reduce_float_t::storage_type*)(rgbs_batch +
                                                                            batch_allocation_size * CDIM);

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

    for (int b = 0; b < num_batches; ++b) {
        block.sync();

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
                float ra = 1.0f / (1.0f - alpha);
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
            warpSum<CDIM, kWaveSize>(v_rgb_local, warp_storage_base);
            warpSum<kWaveSize>(v_conic_local, warp_storage_base);
            warpSum<kWaveSize>(v_xy_local, warp_storage_base);
            warpSum<kWaveSize>(v_opacity_local, warp_storage_base);
            if (warp.thread_rank() == 0) {
                int g = id_batch[t];
                float* v_rgb_ptr = v_colors + CDIM * g;
#pragma unroll
                for (int k = 0; k < CDIM; ++k) {
                    atomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float* v_conic_ptr = (float*)(v_conics) + 3 * g;
                atomicAdd(v_conic_ptr, v_conic_local.x);
                atomicAdd(v_conic_ptr + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float* v_xy_ptr = (float*)(v_means2d) + 2 * g;
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);

                atomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

// How many Gaussians a block stages at once, and the dynamic shared-memory
// bytes that staging plus the rocprim scratch requires. With three color
// channels the batch is the whole 256-thread block.
static int baseline_batch_size(size_t* shmem) {
    const int block_size = kTileSize * kTileSize;
    const int max_batch_size = block_size;
    const int warps_per_block = (block_size + kWaveSize - 1) / kWaveSize;
    const size_t warp_scratch_bytes =
        warps_per_block * sizeof(rocprim::warp_reduce<float, kWaveSize>::storage_type);
    *shmem = max_batch_size * (sizeof(int) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * kCdim) +
             warp_scratch_bytes;
    return max_batch_size;
}

// Launch the baseline backward kernel. Does nothing when the scene has no
// intersections.
void launch_rasterize_baseline(const BwdArgs& args) {
    if (args.n_isects == 0) {
        return;
    }
    size_t shmem = 0;
    const int max_batch_size = baseline_batch_size(&shmem);
    auto kernel = rasterize_to_pixels_3dgs_bwd_baseline<kCdim, float>;
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
