/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (BSD license)
 */
#include "common.h"
#include "tex2d_compat.h"

static void WarpImage(const float *src, int w, int h, int s,
                      const float *u, const float *v, float *out) {

  TEX_CREATE(texToWarp, w, h, s);
  #pragma omp target teams distribute parallel for collapse(2) map(to: texToWarp)
  for (int iy = 0; iy < h; ++iy) {
    for (int ix = 0; ix < w; ++ix) {
      const int pos = ix + iy * s;
      float x = ((float)ix + u[pos] + 0.5f) / (float)w;
      float y = ((float)iy + v[pos] + 0.5f) / (float)h;
      out[pos] = TEX_FETCH(texToWarp, src, x, y);
    }
  }
  TEX_DESTROY(texToWarp);
}
