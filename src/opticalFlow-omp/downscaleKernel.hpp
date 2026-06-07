/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (BSD license)
 */
#include "common.h"
#include "tex2d_compat.h"

static void Downscale(const float *src, int width, int height, int stride,
                      int newWidth, int newHeight, int newStride, float *out) {

  TEX_CREATE(texFine, width, height, stride);

  #pragma omp target teams distribute parallel for collapse(2) map(to:texFine)
  for (int iy = 0; iy < newHeight; ++iy) {
    for (int ix = 0; ix < newWidth; ++ix) {
      float dx = 1.0f / (float)newWidth;
      float dy = 1.0f / (float)newHeight;
      float x = ((float)ix + 0.5f) * dx;
      float y = ((float)iy + 0.5f) * dy;

      out[ix + iy * newStride] = 0.25f * (TEX_FETCH(texFine, src, x - dx*0.25f, y) +
                                   TEX_FETCH(texFine, src, x + dx*0.25f, y) +
                                   TEX_FETCH(texFine, src, x, y - dy*0.25f) +
                                   TEX_FETCH(texFine, src, x, y + dy*0.25f));
    }
  }
  TEX_DESTROY(texFine);
}
