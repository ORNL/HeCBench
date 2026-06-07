/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (BSD license)
 */
#include "common.h"
#include "tex2d_compat.h"
#define TEX_TYPE Tex2DDesc
#define TEX_FETCH(t,x,y) tex2D_sw(t,x,y)
#define TEX_CREATE(v,s,w,h,st) Tex2DDesc v{s,w,h,st}
#define TEX_DESTROY(v)

__global__ void UpscaleKernel(int width, int height, int stride,
                              float scale, float *out, TEX_TYPE texCoarse) {
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;
  if (ix >= width || iy >= height) return;
  float x = ((float)ix + 0.5f) / (float)width;
  float y = ((float)iy + 0.5f) / (float)height;
  out[ix + iy * stride] = TEX_FETCH(texCoarse, x, y) * scale;
}

static void Upscale(const float *src, int width, int height, int stride,
                    int newWidth, int newHeight, int newStride,
                    float scale, float *out) {
  dim3 threads(32, 8);
  dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));
  TEX_CREATE(texCoarse, src, width, height, stride);
  UpscaleKernel<<<blocks, threads>>>(newWidth, newHeight, newStride, scale, out, texCoarse);
  checkCudaErrors(hipDeviceSynchronize());
  TEX_DESTROY(texCoarse);
}

#undef TEX_TYPE
#undef TEX_FETCH
#undef TEX_CREATE
#undef TEX_DESTROY
