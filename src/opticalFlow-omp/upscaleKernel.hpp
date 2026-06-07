/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (BSD license)
 */
#include "common.h"
#include "tex2d_compat.h"

static void Upscale(const float *src, int width, int height, int stride,
                    int newWidth, int newHeight, int newStride,
                    float scale, float *out) {
  TEX_CREATE(texCoarse, width, height, stride);

  #pragma omp target teams distribute parallel for collapse(2) map(to: texCoarse)
  for (int iy = 0; iy < newHeight; ++iy) {
    for (int ix = 0; ix < newWidth; ++ix) {
      float x = ((float)ix + 0.5f) / (float)newWidth;
      float y = ((float)iy + 0.5f) / (float)newHeight;
      out[ix + iy * newStride] = TEX_FETCH(texCoarse, src, x, y) * scale;
    }
  }
  TEX_DESTROY(texCoarse);
}
