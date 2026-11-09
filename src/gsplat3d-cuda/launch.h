#pragma once

#include "types.h"

// Image size and device pointers shared by the forward launch and both
// backward launches. Gradient pointers are read-write; every other pointer
// is an input. render_alphas and last_ids are written by the forward kernel
// and read by the backward kernels.
struct BwdArgs {
    int n;
    int n_isects;
    int image_width;
    int image_height;
    int tile_width;
    int tile_height;
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

// Front-to-back alpha pass. Writes render_alphas and last_ids.
void launch_rasterize_forward(const BwdArgs& args);

// gsplat backward kernel. Lanes cooperate in groups of 32, which is one
// warp. AtomicAdds into the four gradient buffers.
void launch_rasterize_baseline(const BwdArgs& args);

// Same gradients as the baseline, with the warp reduction and shared-memory
// flush described on the kernel.
void launch_rasterize_optimized(const BwdArgs& args);
