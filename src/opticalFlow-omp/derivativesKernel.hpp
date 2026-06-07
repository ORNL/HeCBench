/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (BSD license)
 */
#include "common.h"
#include "tex2d_compat.h"

static void ComputeDerivatives(const float *I0, const float *I1,
                               int w, int h, int s,
                               float *Ix, float *Iy, float *Iz) {
  TEX_CREATE(texSource, w, h, s);
  TEX_CREATE(texTarget, w, h, s);

  #pragma omp target teams distribute parallel for collapse(2) map(to: texSource, texTarget)
  for (int iy = 0; iy < h; ++iy) {
    for (int ix = 0; ix < w; ++ix) {
      const int pos = ix + iy * s;
      float dx = 1.0f / (float)w, dy = 1.0f / (float)h;
      float x = ((float)ix + 0.5f) * dx;
      float y = ((float)iy + 0.5f) * dy;

      float t0, t1;
      t0  = TEX_FETCH(texSource, I0, x-2.f*dx, y);
      t0 -= TEX_FETCH(texSource, I0, x-1.f*dx, y) * 8.f;
      t0 += TEX_FETCH(texSource, I0, x+1.f*dx, y) * 8.f;
      t0 -= TEX_FETCH(texSource, I0, x+2.f*dx, y);
      t0 /= 12.f;

      t1  = TEX_FETCH(texTarget, I1, x-2.f*dx, y);
      t1 -= TEX_FETCH(texTarget, I1, x-1.f*dx, y) * 8.f;
      t1 += TEX_FETCH(texTarget, I1, x+1.f*dx, y) * 8.f;
      t1 -= TEX_FETCH(texTarget, I1, x+2.f*dx, y);
      t1 /= 12.f;
      Ix[pos] = (t0 + t1) * 0.5f;

      Iz[pos] = TEX_FETCH(texTarget, I1, x, y) - TEX_FETCH(texSource, I0, x, y);

      t0  = TEX_FETCH(texSource, I0, x, y-2.f*dy);
      t0 -= TEX_FETCH(texSource, I0, x, y-1.f*dy) * 8.f;
      t0 += TEX_FETCH(texSource, I0, x, y+1.f*dy) * 8.f;
      t0 -= TEX_FETCH(texSource, I0, x, y+2.f*dy);
      t0 /= 12.f;

      t1  = TEX_FETCH(texTarget, I1, x, y-2.f*dy);
      t1 -= TEX_FETCH(texTarget, I1, x, y-1.f*dy) * 8.f;
      t1 += TEX_FETCH(texTarget, I1, x, y+1.f*dy) * 8.f;
      t1 -= TEX_FETCH(texTarget, I1, x, y+2.f*dy);
      t1 /= 12.f;
      Iy[pos] = (t0 + t1) * 0.5f;
    }

  }
  TEX_DESTROY(texSource);
  TEX_DESTROY(texTarget);
}
