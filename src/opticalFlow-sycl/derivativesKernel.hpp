/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (BSD license)
 */
#include "common.h"

#include "tex2d_compat.h"
#define TEX_TYPE Tex2DDesc
#define TEX_FETCH(t,x,y) tex2D_sw(t,x,y)
#define TEX_CREATE(v,s,w,h,st) Tex2DDesc v{s,w,h,st}
#define TEX_DESTROY(v)

void ComputeDerivativesKernel(int width, int height, int stride,
                                         float *Ix, float *Iy, float *Iz,
                                         TEX_TYPE texSource, TEX_TYPE texTarget,
                                         const sycl::nd_item<3> &item) {
  const int ix = item.get_global_id(2);
  const int iy = item.get_global_id(1);
  const int pos = ix + iy * stride;
  if (ix >= width || iy >= height) return;

  float dx = 1.0f/(float)width, dy = 1.0f/(float)height;
  float x  = ((float)ix + 0.5f) * dx;
  float y  = ((float)iy + 0.5f) * dy;

  float t0, t1;
  t0  = TEX_FETCH(texSource, x-2.f*dx, y);
  t0 -= TEX_FETCH(texSource, x-1.f*dx, y) * 8.f;
  t0 += TEX_FETCH(texSource, x+1.f*dx, y) * 8.f;
  t0 -= TEX_FETCH(texSource, x+2.f*dx, y);
  t0 /= 12.f;

  t1  = TEX_FETCH(texTarget, x-2.f*dx, y);
  t1 -= TEX_FETCH(texTarget, x-1.f*dx, y) * 8.f;
  t1 += TEX_FETCH(texTarget, x+1.f*dx, y) * 8.f;
  t1 -= TEX_FETCH(texTarget, x+2.f*dx, y);
  t1 /= 12.f;
  Ix[pos] = (t0 + t1) * 0.5f;

  Iz[pos] = TEX_FETCH(texTarget, x, y) - TEX_FETCH(texSource, x, y);

  t0  = TEX_FETCH(texSource, x, y-2.f*dy);
  t0 -= TEX_FETCH(texSource, x, y-1.f*dy) * 8.f;
  t0 += TEX_FETCH(texSource, x, y+1.f*dy) * 8.f;
  t0 -= TEX_FETCH(texSource, x, y+2.f*dy);
  t0 /= 12.f;

  t1  = TEX_FETCH(texTarget, x, y-2.f*dy);
  t1 -= TEX_FETCH(texTarget, x, y-1.f*dy) * 8.f;
  t1 += TEX_FETCH(texTarget, x, y+1.f*dy) * 8.f;
  t1 -= TEX_FETCH(texTarget, x, y+2.f*dy);
  t1 /= 12.f;
  Iy[pos] = (t0 + t1) * 0.5f;
}

static void ComputeDerivatives(const float *I0, const float *I1,
                               int w, int h, int s,
                               float *Ix, float *Iy, float *Iz,
                               sycl::queue &q)
{
  sycl::range<3> threads(1, 6, 32);
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));

  TEX_CREATE(texSource, I0, w, h, s);
  TEX_CREATE(texTarget, I1, w, h, s);
  q.parallel_for(
      sycl::nd_range<3>(blocks * threads, threads),
      [=](sycl::nd_item<3> item) {
        ComputeDerivativesKernel(w, h, s, Ix, Iy, Iz, texSource, texTarget,
                                 item);
  }).wait();
  TEX_DESTROY(texSource);
  TEX_DESTROY(texTarget);
}

#undef TEX_TYPE
#undef TEX_FETCH
#undef TEX_CREATE
#undef TEX_DESTROY
