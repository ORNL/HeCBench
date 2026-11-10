#pragma once

#include <vector>

#include "types.h"

// Host result of one forward and backward rasterization.
// Gradient buffers are packed the same way as the device downloads:
// means2d n*2, conics n*3, colors n*3, opacities n.
struct Reference {
    std::vector<float> render_alphas;
    std::vector<int> last_ids;
    std::vector<float> means2d;
    std::vector<float> conics;
    std::vector<float> colors;
    std::vector<float> opacities;
};

// Scalar front-to-back composite and its analytic parameter gradients.
// Each pixel is independent. Gradient contributions are summed in double.
Reference reference_rasterize(int n, int width, int height, int tile_w, int tile_h,
                              const std::vector<vec2>& means2d, const std::vector<vec3>& conics,
                              const std::vector<float>& colors, const std::vector<float>& opacities,
                              const std::vector<int>& tile_offsets, const std::vector<int>& flatten_ids,
                              const std::vector<float>& v_render_colors, const std::vector<float>& v_render_alphas);
