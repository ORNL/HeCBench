/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (BSD license)
 */
#include "common.h"
#include "tex2d_compat.h"
#define TEX_TYPE Tex2DDesc
#define TEX_FETCH(t,x,y) tex2D_sw(t,x,y)
#define TEX_CREATE(v,s,w,h,st) Tex2DDesc v{s,w,h,st}
#define TEX_DESTROY(v)

__global__ void WarpingKernel(int width, int height, int stride,
                              const float *u, const float *v,
                              float *out, TEX_TYPE texToWarp) {
  const int ix  = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy  = threadIdx.y + blockIdx.y * blockDim.y;
  const int pos = ix + iy * stride;
  if (ix >= width || iy >= height) return;
  float x = ((float)ix + u[pos] + 0.5f) / (float)width;
  float y = ((float)iy + v[pos] + 0.5f) / (float)height;
  out[pos] = TEX_FETCH(texToWarp, x, y);
}

static void WarpImage(const float *src, int w, int h, int s,
                      const float *u, const float *v, float *out) {
  dim3 threads(32, 6);
  dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
  TEX_CREATE(texToWarp, src, w, h, s);
  WarpingKernel<<<blocks, threads>>>(w, h, s, u, v, out, texToWarp);
  checkCudaErrors(hipDeviceSynchronize());
  TEX_DESTROY(texToWarp);
}

#undef TEX_TYPE
#undef TEX_FETCH
#undef TEX_CREATE
#undef TEX_DESTROY
