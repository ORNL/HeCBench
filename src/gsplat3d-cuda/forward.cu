// Derived from the 3DGS rasterization kernels in gsplat
// (https://github.com/nerfstudio-project/gsplat), Apache-2.0, see LICENSE.
// SPDX-License-Identifier: Apache-2.0

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "launch.h"
#include "raster_math.h"

namespace cg = cooperative_groups;

// Front-to-back alpha pass for one pixel per thread. A 16x16 block owns one
// tile and walks that tile's Gaussians from near to far, staging each batch
// in shared memory. A Gaussian is skipped when sigma is negative or alpha is
// below 1/255, and the walk stops before a Gaussian that would push the
// remaining transmittance to 1e-4 or below.
// render_alphas = 1 - final transmittance. last_ids is the intersection index
// of the last Gaussian that was blended, or 0 when the pixel blends nothing.
// Pixel colors are not stored.
__global__ void rasterize_to_pixels_3dgs_fwd_kernel(
    const int image_width,
    const int image_height,
    const int tile_size,
    const int tile_width,
    const int tile_height,
    const int n_isects,
    const vec2* __restrict__ means2d,
    const vec3* __restrict__ conics,
    const float* __restrict__ opacities,
    const int* __restrict__ tile_offsets,
    const int* __restrict__ flatten_ids,
    float* __restrict__ render_alphas,
    int* __restrict__ last_ids) {
    auto block = cg::this_thread_block();
    const int tile_id = block.group_index().y * tile_width + block.group_index().z;
    const int i = block.group_index().y * tile_size + block.thread_index().y;
    const int j = block.group_index().z * tile_size + block.thread_index().x;
    const bool inside = (i < image_height && j < image_width);
    const float px = j + 0.5f;
    const float py = i + 0.5f;
    const int pix_id = min(i * image_width + j, image_width * image_height - 1);

    bool done = !inside;
    const int range_start = tile_offsets[tile_id];
    const int range_end =
        (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int block_size = block.size();
    const int num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int smem[];
    int* id_batch = smem;
    vec3* xy_opacity_batch = reinterpret_cast<vec3*>(&id_batch[block_size]);
    vec3* conic_batch = reinterpret_cast<vec3*>(&xy_opacity_batch[block_size]);

    float T = 1.0f;
    int cur_idx = 0;
    const int tr = block.thread_rank();

    for (int b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        const int batch_start = range_start + block_size * b;
        const int idx = batch_start + tr;
        if (idx < range_end) {
            const int g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr] = conics[g];
        }
        block.sync();

        const int batch_size = min(block_size, range_end - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const vec3 conic = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
            const GsplatEval eval =
                gsplat_eval(conic.x, conic.y, conic.z, delta.x, delta.y, opac);
            if (eval.sigma < 0.f || eval.alpha < kAlphaThreshold) {
                continue;
            }

            const float next_T = gsplat_next_T(T, eval.alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }

            cur_idx = batch_start + t;
            T = next_T;
        }
    }

    if (inside) {
        render_alphas[pix_id] = gsplat_one_minus(T);
        last_ids[pix_id] = cur_idx;
    }
}

// Launch one 16x16 block per tile. Shared memory holds one batch of Gaussian
// ids, screen-space centers with opacity, and conics.
void launch_rasterize_forward(const BwdArgs& args) {
    const dim3 threads(kTileSize, kTileSize, 1);
    const dim3 grid(1, args.tile_height, args.tile_width);
    const int block_size = kTileSize * kTileSize;
    const size_t shmem = block_size * (sizeof(int) + sizeof(vec3) + sizeof(vec3));
    rasterize_to_pixels_3dgs_fwd_kernel<<<grid, threads, shmem>>>(
        args.image_width,
        args.image_height,
        kTileSize,
        args.tile_width,
        args.tile_height,
        args.n_isects,
        args.means2d,
        args.conics,
        args.opacities,
        args.tile_offsets,
        args.flatten_ids,
        const_cast<float*>(args.render_alphas),
        const_cast<int*>(args.last_ids));
}
