#ifndef RMSNORM_QUANT_REFERENCE_H
#define RMSNORM_QUANT_REFERENCE_H

#include "fp8_e4m3.h"
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>

inline void rmsnorm_quant_cpu(uint8_t *out, const float *inp, const float *gamma,
                              float scale, bool invert, bool fnuz, int64_t N,
                              int64_t H, float eps = 1e-5f) {
  for (int64_t t = 0; t < N; t++) {
    const float *x = inp + t * H;
    float m = 0.0f;
    for (int64_t i = 0; i < H; i++)
      m += x[i] * x[i];
    m = m / static_cast<float>(H);
    const float s = 1.0f / sqrtf(m + eps);
    uint8_t *out_t = out + t * H;
    for (int64_t i = 0; i < H; i++) {
      const float o = x[i] * s * gamma[i];
      out_t[i] = rq::scaled_fp8(o, scale, invert, fnuz);
    }
  }
}

// Return false if any code differs from the CPU reference by more than max_diff.
inline bool check_fp8_vs_cpu(const uint8_t *got, const uint8_t *ref, size_t n,
                             bool fnuz, int max_diff) {
  int nfaults = 0;
  for (size_t j = 0; j < n; j++) {
    const int d = abs((int)got[j] - (int)ref[j]);
    if (d > max_diff) {
      printf("Mismatch at %zu: CPU %02x GPU %02x (dequant CPU %.6f GPU %.6f)\n",
             j, ref[j], got[j], rq::fp8_e4m3_to_f32(ref[j], fnuz),
             rq::fp8_e4m3_to_f32(got[j], fnuz));
      if (++nfaults >= 10)
        return false;
    }
  }
  return nfaults == 0;
}

#endif
