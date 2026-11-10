// Derived from the 3DGS rasterization kernels in gsplat
// (https://github.com/nerfstudio-project/gsplat), Apache-2.0, see LICENSE.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdlib>

#include <sycl/sycl.hpp>

#include "launch.h"
#include "raster_math.h"

// Front-to-back alpha pass for one pixel per thread. A 16x16 work-group owns
// one tile and walks that tile's Gaussians from near to far, staging each
// batch in local memory. A Gaussian is skipped when sigma is negative or
// alpha is below 1/255, and the walk stops before a Gaussian that would push
// the remaining transmittance to 1e-4 or below.
// render_alphas = 1 - final transmittance. last_ids is the intersection index
// of the last Gaussian that was blended, or 0 when the pixel blends nothing.
// Pixel colors are not stored.
static void rasterize_to_pixels_3dgs_fwd_kernel(
    const sycl::nd_item<3>& item,
    int* smem,
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
    auto block = item.get_group();
    const int tile_id = item.get_group(1) * tile_width + item.get_group(2);
    const int i = item.get_group(1) * tile_size + item.get_local_id(1);
    const int j = item.get_group(2) * tile_size + item.get_local_id(2);
    const bool inside = (i < image_height && j < image_width);
    const float px = j + 0.5f;
    const float py = i + 0.5f;
    const int pix_id = sycl::min(i * image_width + j, image_width * image_height - 1);

    bool done = !inside;
    const int range_start = tile_offsets[tile_id];
    const int range_end =
        (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int block_size = item.get_local_range().size();
    const int num_batches = (range_end - range_start + block_size - 1) / block_size;

    int* id_batch = smem;
    vec3* xy_opacity_batch = reinterpret_cast<vec3*>(&id_batch[block_size]);
    vec3* conic_batch = reinterpret_cast<vec3*>(&xy_opacity_batch[block_size]);

    float T = 1.0f;
    int cur_idx = 0;
    const int tr = item.get_local_linear_id();

    for (int b = 0; b < num_batches; ++b) {
        // The barrier orders the previous batch's reads before this batch's
        // writes; all_of_group alone does not promise that.
        sycl::group_barrier(block);
        if (sycl::all_of_group(block, done)) {
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
        sycl::group_barrier(block);

        const int batch_size = sycl::min(block_size, range_end - batch_start);
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

// Launch one 16x16 work-group per tile. Local memory holds one batch of
// Gaussian ids, screen-space centers with opacity, and conics.
void launch_rasterize_forward(sycl::queue& q, const BwdArgs& args) {
    const sycl::range<3> threads(1, kTileSize, kTileSize);
    const sycl::range<3> grid(1, args.tile_height * kTileSize, args.tile_width * kTileSize);
    const int block_size = kTileSize * kTileSize;
    const size_t shmem = block_size * (sizeof(int) + sizeof(vec3) + sizeof(vec3));
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> smem(sycl::range<1>(shmem / sizeof(int)), cgh);
        cgh.parallel_for(sycl::nd_range<3>(grid, threads), [=](sycl::nd_item<3> item) {
            rasterize_to_pixels_3dgs_fwd_kernel(
                item,
                smem.get_multi_ptr<sycl::access::decorated::no>().get(),
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
        });
    });
}

int select_sub_group_size(const sycl::queue& q) {
    bool has32 = false;
    bool has64 = false;
    for (size_t s : q.get_device().get_info<sycl::info::device::sub_group_sizes>()) {
        has32 = has32 || s == 32;
        has64 = has64 || s == 64;
    }
    if (has32) return 32;
    if (has64) return 64;
    fprintf(stderr, "SYCL error: the device supports neither sub-group size 32 nor 64\n");
    exit(EXIT_FAILURE);
}
