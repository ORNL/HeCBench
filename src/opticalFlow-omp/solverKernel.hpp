/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 */
#include "common.h"

static void SolveForUpdate(const float *du0, const float *dv0, const float *Ix,
                           const float *Iy, const float *Iz, int w,
                           int h, int s, float alpha, float *du1, float *dv1) {
  #pragma omp target teams distribute parallel for collapse(2)
  for (int iy = 0; iy < h; ++iy) {
    for (int ix = 0; ix < w; ++ix) {
      const int pos = iy * s + ix;

      // Clamp-to-edge neighbour indices
      const int x_left  = std::max(ix - 1, 0);
      const int x_right = std::min(ix + 1, w - 1);
      const int y_up    = std::min(iy + 1, h - 1);
      const int y_down  = std::max(iy - 1, 0);

      // Neighbour linear addresses
      const int pos_left  = iy      * s + x_left;
      const int pos_right = iy      * s + x_right;
      const int pos_up    = y_up    * s + ix;
      const int pos_down  = y_down  * s + ix;

      // Jacobi stencil
      const float sumU = (du0[pos_left] + du0[pos_right] +
                          du0[pos_up]   + du0[pos_down]) * 0.25f;
      const float sumV = (dv0[pos_left] + dv0[pos_right] +
                          dv0[pos_up]   + dv0[pos_down]) * 0.25f;

      const float frac = (Ix[pos] * sumU + Iy[pos] * sumV + Iz[pos]) /
                         (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

      du1[pos] = sumU - Ix[pos] * frac;
      dv1[pos] = sumV - Iy[pos] * frac;
    }
  }
}
