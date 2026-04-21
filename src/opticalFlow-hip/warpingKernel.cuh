/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief warp image with a given displacement field, CUDA kernel.
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[in]  u       horizontal displacement
/// \param[in]  v       vertical displacement
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
__global__ void WarpingKernel(int width, int height, int stride, const float *u,
                              const float *v, float *out,
                              hipTextureObject_t texToWarp) {
  const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const int iy = threadIdx.y + blockIdx.y * blockDim.y;

  const int pos = ix + iy * stride;

  if (ix >= width || iy >= height) return;

  float x = ((float)ix + u[pos] + 0.5f) / (float)width;
  float y = ((float)iy + v[pos] + 0.5f) / (float)height;

  out[pos] = tex2D<float>(texToWarp, x, y);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief warp image with provided vector field, CUDA kernel wrapper.
///
/// For each output pixel there is a vector which tells which pixel
/// from a source image should be mapped to this particular output
/// pixel.
/// It is assumed that images and the vector field have the same stride and
/// resolution.
/// \param[in]  src source image
/// \param[in]  w   width
/// \param[in]  h   height
/// \param[in]  s   stride
/// \param[in]  u   horizontal displacement
/// \param[in]  v   vertical displacement
/// \param[out] out warped image
///////////////////////////////////////////////////////////////////////////////
static void WarpImage(const float *src, int w, int h, int s, const float *u,
                      const float *v, float *out) {
  dim3 threads(32, 6);
  dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

  hipResourceDesc texRes;
  memset(&texRes, 0, sizeof(hipResourceDesc));

#ifdef __HIP_PLATFORM_AMD__
  // AMD: Pitch2D + normalizedCoords/linear filter is unsupported; stage into a hipArray
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();
  hipArray_t srcArray;
  checkCudaErrors(hipMallocArray(&srcArray, &channelDesc, w, h));
  checkCudaErrors(hipMemcpy2DToArray(srcArray, 0, 0, src,
                                     s * sizeof(float),
                                     w * sizeof(float), h,
                                     hipMemcpyDeviceToDevice));
  texRes.resType = hipResourceTypeArray;
  texRes.res.array.array = srcArray;
#else
  texRes.resType = hipResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = (void *)src;
  texRes.res.pitch2D.desc = hipCreateChannelDesc<float>();
  texRes.res.pitch2D.width = w;
  texRes.res.pitch2D.height = h;
  texRes.res.pitch2D.pitchInBytes = s * sizeof(float);
#endif

  hipTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(hipTextureDesc));

  texDescr.normalizedCoords = true;
  texDescr.filterMode = hipFilterModeLinear;
  texDescr.addressMode[0] = hipAddressModeMirror;
  texDescr.addressMode[1] = hipAddressModeMirror;
  texDescr.readMode = hipReadModeElementType;

  hipTextureObject_t texToWarp;
  checkCudaErrors(
      hipCreateTextureObject(&texToWarp, &texRes, &texDescr, NULL));

  WarpingKernel<<<blocks, threads>>>(w, h, s, u, v, out, texToWarp);

  checkCudaErrors(hipDestroyTextureObject(texToWarp));
#ifdef __HIP_PLATFORM_AMD__
  checkCudaErrors(hipFreeArray(srcArray));
#endif
}
