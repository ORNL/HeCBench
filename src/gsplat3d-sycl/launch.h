#pragma once

#include <sycl/sycl.hpp>

#include "types.h"

// Image size and device pointers shared by the forward launch and both
// backward launches. Gradient pointers are read-write; every other pointer
// is an input. render_alphas and last_ids are written by the forward kernel
// and read by the backward kernels. sub_group_size is 32 or 64 and selects
// which instantiation of each backward kernel runs.
struct BwdArgs {
    int n;
    int n_isects;
    int image_width;
    int image_height;
    int tile_width;
    int tile_height;
    int sub_group_size;
    const vec2* means2d;
    const vec3* conics;
    const float* colors;
    const float* opacities;
    const int* tile_offsets;
    const int* flatten_ids;
    const float* render_alphas;
    const int* last_ids;
    const float* v_render_colors;
    const float* v_render_alphas;
    vec2* v_means2d;
    vec3* v_conics;
    float* v_colors;
    float* v_opacities;
};

// Sub-group size the backward kernels run with on this device: 32 when the
// device supports it, otherwise 64. Aborts when it supports neither.
int select_sub_group_size(const sycl::queue& q);

// Front-to-back alpha pass. Writes render_alphas and last_ids.
void launch_rasterize_forward(sycl::queue& q, const BwdArgs& args);

// gsplat backward kernel. Lanes cooperate in groups of 32, which is a whole
// sub-group of 32 or half of a sub-group of 64. AtomicAdds into the four
// gradient buffers.
void launch_rasterize_baseline(sycl::queue& q, const BwdArgs& args);

// Same gradients as the baseline, with the group reduction and local-memory
// flush described on the kernel.
void launch_rasterize_optimized(sycl::queue& q, const BwdArgs& args);
