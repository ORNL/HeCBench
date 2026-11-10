#include <cstddef>
#include <vector>

#include "raster_math.h"
#include "reference.h"

namespace {

// One Gaussian that a pixel actually blended, with the transmittance that
// reached it and the geometry used to differentiate alpha.
struct Sample {
    int id;
    float T;
    float alpha;
    float vis;
    float opac;
    float dx;
    float dy;
    vec3 conic;
};

}  // namespace

// Host oracle for one forward and backward rasterization.
//
// Front to back, a pixel composites the Gaussians of its tile in depth order:
//   delta = mean - pixel_center
//   sigma = 0.5 * (c_x dx^2 + c_z dy^2) + c_y dx dy
//   alpha = min(0.999, opacity * exp(-sigma))
// Samples with sigma < 0 or alpha < 1/255 are skipped. The walk also stops
// before a sample that would take the transmittance to 1e-4 or below.
//   weight_i = alpha_i * T_i
//   T_{i+1}  = T_i * (1 - alpha_i),  T_0 = 1
//   render_alpha = 1 - T_final
//
// The color of a pixel is sum_i color_i * weight_i. Differentiating that sum
// and the final transmittance with respect to alpha_i gives
//   dC_k / d alpha_i = color_{i,k} * T_i - sum_{j>i} color_{j,k} * weight_j / (1 - alpha_i)
//   dA   / d alpha_i = T_final / (1 - alpha_i)
// The sum over later samples is kept as a running color, updated back to front.
// Opacity, mean, and conic gradients flow only while opacity * exp(-sigma)
// is at or below the 0.999 clamp, where alpha still depends on them.
// Contributions are accumulated in double. Each sample keeps the transmittance
// from this forward product, rather than the value recovered by dividing
// 1 - render_alpha on the way back.
Reference reference_rasterize(int n, int width, int height, int tile_w, int tile_h,
                              const std::vector<vec2>& means2d, const std::vector<vec3>& conics,
                              const std::vector<float>& colors, const std::vector<float>& opacities,
                              const std::vector<int>& tile_offsets, const std::vector<int>& flatten_ids,
                              const std::vector<float>& v_render_colors, const std::vector<float>& v_render_alphas) {
    const int tiles = tile_w * tile_h;
    const int n_isects = static_cast<int>(flatten_ids.size());

    std::vector<double> acc_means((size_t)n * 2, 0.0);
    std::vector<double> acc_conics((size_t)n * 3, 0.0);
    std::vector<double> acc_colors((size_t)n * kCdim, 0.0);
    std::vector<double> acc_opac((size_t)n, 0.0);

    Reference out;
    out.render_alphas.assign((size_t)width * height, 0.0f);
    out.last_ids.assign((size_t)width * height, 0);

    std::vector<Sample> samples;
    samples.reserve(64);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int tile = (y / kTileSize) * tile_w + (x / kTileSize);
            const int range_start = tile_offsets[tile];
            const int range_end = (tile + 1 == tiles) ? n_isects : tile_offsets[tile + 1];
            const size_t pix = (size_t)y * width + x;
            const float px = x + 0.5f;
            const float py = y + 0.5f;

            samples.clear();
            float T = 1.0f;
            int last = 0;
            for (int idx = range_start; idx < range_end; ++idx) {
                const int g = flatten_ids[idx];
                const vec2 mean = means2d[g];
                const vec3 conic = conics[g];
                const float dx = mean.x - px;
                const float dy = mean.y - py;
                const float opac = opacities[g];
                const GsplatEval eval = gsplat_eval(conic.x, conic.y, conic.z, dx, dy, opac);
                if (eval.sigma < 0.0f || eval.alpha < kAlphaThreshold) {
                    continue;
                }
                const float next_T = gsplat_next_T(T, eval.alpha);
                if (next_T <= 1.0e-4f) {
                    break;
                }
                samples.push_back(Sample{g, T, eval.alpha, eval.vis, opac, dx, dy, conic});
                last = idx;
                T = next_T;
            }

            out.render_alphas[pix] = gsplat_one_minus(T);
            out.last_ids[pix] = last;
            const float T_final = T;
            const float* v_color = v_render_colors.data() + pix * kCdim;
            const float v_alpha_pix = v_render_alphas[pix];

            float behind[kCdim] = {0.0f, 0.0f, 0.0f};
            for (size_t s = samples.size(); s-- > 0;) {
                const Sample& sample = samples[s];
                const float ra = 1.0f / (1.0f - sample.alpha);
                const float weight = sample.alpha * sample.T;
                const float* color = colors.data() + (size_t)sample.id * kCdim;

                float v_alpha = T_final * ra * v_alpha_pix;
                for (int k = 0; k < kCdim; ++k) {
                    acc_colors[(size_t)sample.id * kCdim + k] += weight * v_color[k];
                    v_alpha += (color[k] * sample.T - behind[k] * ra) * v_color[k];
                    behind[k] += color[k] * weight;
                }

                const float opac_vis = gsplat_mul(sample.opac, sample.vis);
                if (opac_vis > 0.999f) {
                    continue;
                }
                const float v_sigma = -opac_vis * v_alpha;
                const float vx = v_sigma * (sample.conic.x * sample.dx + sample.conic.y * sample.dy);
                const float vy = v_sigma * (sample.conic.y * sample.dx + sample.conic.z * sample.dy);
                acc_means[(size_t)sample.id * 2 + 0] += vx;
                acc_means[(size_t)sample.id * 2 + 1] += vy;
                acc_conics[(size_t)sample.id * 3 + 0] += 0.5f * v_sigma * sample.dx * sample.dx;
                acc_conics[(size_t)sample.id * 3 + 1] += v_sigma * sample.dx * sample.dy;
                acc_conics[(size_t)sample.id * 3 + 2] += 0.5f * v_sigma * sample.dy * sample.dy;
                acc_opac[sample.id] += sample.vis * v_alpha;
            }
        }
    }

    out.means2d.resize(acc_means.size());
    out.conics.resize(acc_conics.size());
    out.colors.resize(acc_colors.size());
    out.opacities.resize(acc_opac.size());
    for (size_t i = 0; i < acc_means.size(); ++i) out.means2d[i] = static_cast<float>(acc_means[i]);
    for (size_t i = 0; i < acc_conics.size(); ++i) out.conics[i] = static_cast<float>(acc_conics[i]);
    for (size_t i = 0; i < acc_colors.size(); ++i) out.colors[i] = static_cast<float>(acc_colors[i]);
    for (size_t i = 0; i < acc_opac.size(); ++i) out.opacities[i] = static_cast<float>(acc_opac[i]);
    return out;
}
