/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. (BSD license)
 */
#include "common.h"

#include "tex2d_compat.h"
#define TEX_TYPE   Tex2DDesc
#define TEX_FETCH(tex, x, y) tex2D_sw(tex, x, y)
#define TEX_CREATE(texVar, src, width, height, stride) \
    Tex2DDesc texVar { src, width, height, stride }
#define TEX_DESTROY(texVar) /* no-op: no object to destroy */

__global__ void DownscaleKernel(int width, int height, int stride,
                                float *out, TEX_TYPE texFine) {
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;
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
                      int newWidth, int newHeight, int newStride, float *out) {
  dim3 threads(32, 8);
  dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));
  TEX_CREATE(texFine, src, width, height, stride);
  DownscaleKernel<<<blocks, threads>>>(newWidth, newHeight, newStride, out, texFine);
  checkCudaErrors(cudaDeviceSynchronize());
  TEX_DESTROY(texFine);
}

#undef TEX_TYPE
#undef TEX_FETCH
#undef TEX_CREATE
#undef TEX_DESTROY
