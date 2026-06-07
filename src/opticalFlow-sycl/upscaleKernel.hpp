/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (BSD license)
 */
#include "common.h"
#include "tex2d_compat.h"
#define TEX_TYPE Tex2DDesc
#define TEX_FETCH(t,x,y) tex2D_sw(t,x,y)
#define TEX_CREATE(v,s,w,h,st) Tex2DDesc v{s,w,h,st}
#define TEX_DESTROY(v)

void UpscaleKernel(int width, int height, int stride,
                              float scale, float *out, TEX_TYPE texCoarse,
                              const sycl::nd_item<3> &item) {
  const int ix = item.get_global_id(2);
  const int iy = item.get_global_id(1);
  if (ix >= width || iy >= height) return;
  float x = ((float)ix + 0.5f) / (float)width;
  float y = ((float)iy + 0.5f) / (float)height;
  out[ix + iy * stride] = TEX_FETCH(texCoarse, x, y) * scale;
}

static void Upscale(const float *src, int width, int height, int stride,
                    int newWidth, int newHeight, int newStride,
                    float scale, float *out, sycl::queue &q) {
  sycl::range<3> threads(1, 8, 32);
  sycl::range<3> blocks(1, iDivUp(newHeight, threads[1]),
                        iDivUp(newWidth, threads[2]));
  TEX_CREATE(texCoarse, src, width, height, stride);
  q.parallel_for(
    sycl::nd_range<3>(blocks * threads, threads),
    [=](sycl::nd_item<3> item) {
      UpscaleKernel(newWidth, newHeight, newStride, scale, out, texCoarse, item);
  }).wait();
  TEX_DESTROY(texCoarse);
}

#undef TEX_TYPE
#undef TEX_FETCH
#undef TEX_CREATE
#undef TEX_DESTROY
