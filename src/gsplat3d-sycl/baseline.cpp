// Derived from the 3DGS rasterization kernels in gsplat
// (https://github.com/nerfstudio-project/gsplat), Apache-2.0, see LICENSE.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdlib>

#include <sycl/sycl.hpp>

#include "launch.h"
#include "raster_math.h"
#include "warp_sum.h"

// gsplat backward kernel.
// One 16x16 work-group per tile walks that tile's Gaussians from far to near.
// Lanes cooperate in groups of kWaveSize. On a 32-wide sub-group that group
// is the whole sub-group. On a 64-wide sub-group the two halves are separate
// groups. Transmittance starts at 1 - render_alpha and is rebuilt by
// multiplying by 1/(1 - alpha) at each blended Gaussian. The farthest
// intersection any lane in the group still needs is broadcast to every lane
// in that group, so the walk stays convergent. Each group sums the nine
// gradient channels with sub-group shuffles, and lane 0 atomicAdds that sum
// into the global gradients. Mean, conic, and opacity gradients are zero
// where opacity * exp(-sigma) is above the 0.999 clamp.
template <int CDIM, typename scalar_t>
void rasterize_to_pixels_3dgs_bwd_baseline(
    const sycl::nd_item<3>& item,
    int* s,
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
    const int batch_allocation_size = max_batch_size;
    const int num_batches =
        (range_end - range_start + max_batch_size - 1) / max_batch_size;

    int* id_batch = s;
    vec3* xy_opacity_batch = reinterpret_cast<vec3*>(&id_batch[batch_allocation_size]);
    vec3* conic_batch = reinterpret_cast<vec3*>(&xy_opacity_batch[batch_allocation_size]);
    float* rgbs_batch = (float*)&conic_batch[batch_allocation_size];

    auto* warp_storage_base = (warp_reduce_storage*)(rgbs_batch + batch_allocation_size * CDIM);

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

    for (int b = 0; b < num_batches; ++b) {
        sycl::group_barrier(block);

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
            warpSum<CDIM, kWaveSize>(sg, v_rgb_local, warp_storage_base);
            warpSum<kWaveSize>(sg, v_conic_local, warp_storage_base);
            warpSum<kWaveSize>(sg, v_xy_local, warp_storage_base);
            warpSum<kWaveSize>(sg, v_opacity_local, warp_storage_base);
            if (logical_warp_lane(sg) == 0) {
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

// How many Gaussians a work-group stages at once, and the local-memory bytes
// that staging plus the per-group scratch requires. With three color channels
// the batch is the whole 256-item work-group.
static int baseline_batch_size(size_t* shmem) {
    const int block_size = kTileSize * kTileSize;
    const int max_batch_size = block_size;
    const int warps_per_block = (block_size + kWaveSize - 1) / kWaveSize;
    const size_t warp_scratch_bytes = warps_per_block * sizeof(warp_reduce_storage);
    *shmem = max_batch_size * (sizeof(int) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * kCdim) +
             warp_scratch_bytes;
    return max_batch_size;
}

template <int SG>
static void launch_rasterize_baseline_sg(sycl::queue& q, const BwdArgs& args, const int max_batch_size,
                                         const size_t shmem) {
    const sycl::range<3> threads(1, kTileSize, kTileSize);
    const sycl::range<3> grid(1, args.tile_height * kTileSize, args.tile_width * kTileSize);
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::int4, 1> smem(sycl::range<1>((shmem + 15) / 16), cgh);
        cgh.parallel_for(sycl::nd_range<3>(grid, threads), [=](sycl::nd_item<3> item)
                                                               [[sycl::reqd_sub_group_size(SG)]] {
            rasterize_to_pixels_3dgs_bwd_baseline<kCdim, float>(
                item,
                reinterpret_cast<int*>(smem.get_multi_ptr<sycl::access::decorated::no>().get()),
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

// Launch the baseline backward kernel. Does nothing when the scene has no
// intersections. Aborts if the local memory baseline_batch_size computed
// exceeds the device limit.
void launch_rasterize_baseline(sycl::queue& q, const BwdArgs& args) {
    if (args.n_isects == 0) {
        return;
    }
    size_t shmem = 0;
    const int max_batch_size = baseline_batch_size(&shmem);
    const size_t local_mem = q.get_device().get_info<sycl::info::device::local_mem_size>();
    if (shmem > local_mem) {
        fprintf(stderr, "SYCL error %s:%d local memory: need %zu bytes, device has %zu\n", __FILE__, __LINE__,
                shmem, local_mem);
        exit(EXIT_FAILURE);
    }
    if (args.sub_group_size == 32) {
        launch_rasterize_baseline_sg<32>(q, args, max_batch_size, shmem);
    } else {
        launch_rasterize_baseline_sg<64>(q, args, max_batch_size, shmem);
    }
}
