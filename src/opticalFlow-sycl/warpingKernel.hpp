/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (BSD license)
 */
#include "common.h"
#include "tex2d_compat.h"
#define TEX_TYPE Tex2DDesc
#define TEX_FETCH(t,x,y) tex2D_sw(t,x,y)
#define TEX_CREATE(v,s,w,h,st) Tex2DDesc v{s,w,h,st}
#define TEX_DESTROY(v)

void WarpingKernel(int width, int height, int stride,
                              const float *u, const float *v,
                              float *out, TEX_TYPE texToWarp,
                              const sycl::nd_item<3> &item) {
  const int ix = item.get_global_id(2);
  const int iy = item.get_global_id(1);
  const int pos = ix + iy * stride;
  if (ix >= width || iy >= height) return;
  float x = ((float)ix + u[pos] + 0.5f) / (float)width;
  float y = ((float)iy + v[pos] + 0.5f) / (float)height;
  out[pos] = TEX_FETCH(texToWarp, x, y);
}

static void WarpImage(const float *src, int w, int h, int s,
                      const float *u, const float *v, float *out, sycl::queue &q) {
  sycl::range<3> threads(1, 6, 32);
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));
  TEX_CREATE(texToWarp, src, w, h, s);
  q.parallel_for(
      sycl::nd_range<3>(blocks * threads, threads),
      [=](sycl::nd_item<3> item) {
        WarpingKernel(w, h, s, u, v, out, texToWarp, item);
  }).wait();
  TEX_DESTROY(texToWarp);
}

#undef TEX_TYPE
#undef TEX_FETCH
#undef TEX_CREATE
#undef TEX_DESTROY
