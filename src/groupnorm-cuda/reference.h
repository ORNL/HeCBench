#include <cmath>

// -----------------------------------------------------------------------------
// Group Normalization Forward Pass based on the GPU kernel
//   x, out : [B, C, img_size]   (C = n_groups * group_size)
//   weight  : [C]               (per-channel scale, γ)
//   bias    : [C]               (per-channel shift,  β)
//   mean    : [B, n_groups]     (optional, can be nullptr)
//   rstd    : [B, n_groups]     (optional, can be nullptr)
// -----------------------------------------------------------------------------
void groupnorm_forward_ref(
    const float* x,
    const float* weight,
    const float* bias,
    float*       out,
    float*       mean,       // may be nullptr
    float*       rstd,       // may be nullptr
    int B, int C, int img_size, int n_groups
) {
    int group_size = C / n_groups;

    const int group_pixels = img_size * group_size;   // pixels per group per image
    const float eps = 1e-5f;

    for (int b = 0; b < B; b++) {
        for (int g = 0; g < n_groups; g++) {
            int block_idx = b * n_groups + g;

            const float* x_block      = x   + block_idx * group_pixels;
                  float* out_block     = out  + block_idx * group_pixels;
            const float* weight_group  = weight + g * group_size;
            const float* bias_group    = bias   + g * group_size;

            float sum  = 0.0f;
            float sum2 = 0.0f;
            for (int i = 0; i < group_pixels; i++) {
                float val = x_block[i];
                sum  += val;
                sum2 += val * val;
            }

            float m   = sum  / group_pixels;
            float m2  = sum2 / group_pixels;
            float var = m2 - m * m;                // E[x²] - E[x]²
            float s   = 1.0f / std::sqrt(var + eps);  // rsqrtf equivalent

            if (mean != nullptr) mean[block_idx] = m;
            if (rstd != nullptr) rstd[block_idx] = s;

            for (int i = 0; i < group_pixels; i++) {
                int c = i / img_size;
                float n   = s * (x_block[i] - m);
                out_block[i] = n * weight_group[c] + bias_group[c];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Group Normalization Backward Pass based on the GPU kernel
//
//   dout, x, dx : [B, C, img_size]        (C = n_groups * group_size)
//   mean, rstd  : [B * n_groups]
//   weight      : [C]
//   dweight     : [C]   (accumulated, must be zeroed before call)
//   dbias       : [C]   (accumulated, must be zeroed before call)
// -----------------------------------------------------------------------------
void groupnorm_backward_ref(
    const float* dout,
    const float* x,
    const float* mean,
    const float* rstd,
    const float* weight,
    float*       dx,
    float*       dweight,    // zeroed by caller — kernel uses atomicAdd
    float*       dbias,      // zeroed by caller — kernel uses atomicAdd
    int B, int C, int img_size, int n_groups
) {
    int group_size = C / n_groups;

    const int group_pixels = img_size * group_size;

    for (int b = 0; b < B; b++) {
        for (int g = 0; g < n_groups; g++) {
            int block_idx = b * n_groups + g;

            const float* dout_block  = dout   + block_idx * group_pixels;
            const float* x_block     = x      + block_idx * group_pixels;
                  float* dx_block    = dx     + block_idx * group_pixels;
            const float* weight_g    = weight  + g * group_size;
                  float* dweight_g   = dweight + g * group_size;
                  float* dbias_g     = dbias   + g * group_size;

            float m_val    = mean[block_idx];
            float rstd_val = rstd[block_idx];

            float w_dout_sum      = 0.0f;
            float w_dout_norm_sum = 0.0f;

            for (int i = 0; i < group_pixels; i++) {
                int   c = i / img_size;
                float cur_w_dout   = weight_g[c] * dout_block[i];
                w_dout_sum        += cur_w_dout;
                float norm         = (x_block[i] - m_val) * rstd_val;
                w_dout_norm_sum   += cur_w_dout * norm;
            }

            float w_dout_block      = w_dout_sum      / group_pixels;
            float w_dout_norm_block = w_dout_norm_sum / group_pixels;

            // compute dx
            for (int i = 0; i < group_pixels; i++) {
                int   c = i / img_size;
                float dout_val    = dout_block[i];
                float norm        = (x_block[i] - m_val) * rstd_val;
                float w_dout      = weight_g[c] * dout_val;
                dx_block[i]       = (w_dout - w_dout_block - norm * w_dout_norm_block) * rstd_val;
            }

            // compute dweight and dbias
            for (int c = 0; c < group_size; c++) {
                const float* dout_ch = dout_block + c * img_size;
                const float* x_ch   = x_block    + c * img_size;

                float dw = 0.0f;
                float db = 0.0f;
                for (int i = 0; i < img_size; i++) {
                    float dout_val  = dout_ch[i];
                    db             += dout_val;
                    float norm      = (x_ch[i] - m_val) * rstd_val;
                    dw             += dout_val * norm;
                }
                dweight_g[c] += dw;
                dbias_g[c]   += db;
            }
        }
    }
}

