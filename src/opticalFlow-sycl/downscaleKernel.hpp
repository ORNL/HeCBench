/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (BSD license)
 */
#include "common.h"

#include "tex2d_compat.h"
#define TEX_TYPE   Tex2DDesc
#define TEX_FETCH(tex, x, y) tex2D_sw(tex, x, y)
#define TEX_CREATE(texVar, src, width, height, stride) \
    Tex2DDesc texVar { src, width, height, stride }
#define TEX_DESTROY(texVar) /* no-op: no object to destroy */

void DownscaleKernel(int width, int height, int stride,
                                float *out, TEX_TYPE texFine,
                                const sycl::nd_item<3> &item) {
  const int ix = item.get_global_id(2);
  const int iy = item.get_global_id(1);
  if (ix >= width || iy >= height) return;

  float dx = 1.0f / (float)width;
  float dy = 1.0f / (float)height;
  float x  = ((float)ix + 0.5f) * dx;
  float y  = ((float)iy + 0.5f) * dy;

  out[ix + iy * stride] = 0.25f * (TEX_FETCH(texFine, x - dx*0.25f, y) +
                                   TEX_FETCH(texFine, x + dx*0.25f, y) +
                                   TEX_FETCH(texFine, x, y - dy*0.25f) +
                                   TEX_FETCH(texFine, x, y + dy*0.25f));
}

static void Downscale(const float *src, int width, int height, int stride,
                      int newWidth, int newHeight, int newStride, float *out,
                      sycl::queue &q)
{
  sycl::range<3> threads(1, 8, 32);
  sycl::range<3> blocks(1, iDivUp(newHeight, threads[1]),
                        iDivUp(newWidth, threads[2]));
  TEX_CREATE(texFine, src, width, height, stride);

  q.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
    [=](sycl::nd_item<3> item) {
      DownscaleKernel(newWidth, newHeight, newStride, out, texFine, item);
  }).wait();
  TEX_DESTROY(texFine);
}

#undef TEX_TYPE
#undef TEX_FETCH
#undef TEX_CREATE
#undef TEX_DESTROY
