/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <sycl/sycl.hpp>
#include "shrUtils.h"
#include <cstdlib>

extern
void BoxFilterHost( unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage,
                    int uiWidth, int uiHeight, int r, float fScale );


const unsigned int RADIUS = 10;                    // initial radius of 2D box filter mask
const float SCALE = 1.0f/(2.0f * RADIUS + 1.0f);  // precalculated GV rescaling value

inline uint DivUp(const uint a, const uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Helper function to convert float[4] rgba color to 32-bit unsigned integer
sycl::float4 rgbaUintToFloat4(const unsigned int c)
{
    sycl::float4 rgba;
    rgba.x() = c & 0xff;
    rgba.y() = (c >> 8) & 0xff;
    rgba.z() = (c >> 16) & 0xff;
    rgba.w() = (c >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
unsigned int rgbaFloat4ToUint(const sycl::float4 rgba, const float fScale)
{
    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x() * fScale);
    uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y() * fScale)) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z() * fScale)) << 16);
    uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w() * fScale)) << 24);
    return uiPackedPix;
}

void BoxFilterGPU ( sycl::queue &q,
                    sycl::uchar4 *cmBufIn,
                    unsigned int *cmBufTmp,
                    unsigned int *cmBufOut,
                    const int uiWidth,
                    const int uiHeight,
                    const int iRadius,
                    const float fScale,
                    const int iCycles)
{
    const int szMaxWorkgroupSize = 256;
    const int iRadiusAligned = ((iRadius + 15)/16) * 16;  // 16
    unsigned int uiNumOutputPix = 64;  // Default output pix per workgroup
    if (szMaxWorkgroupSize < (iRadiusAligned + uiNumOutputPix + iRadius))
      uiNumOutputPix = szMaxWorkgroupSize - iRadiusAligned - iRadius;

    // Set global and local work sizes for row kernel // Workgroup padded left and right
    sycl::range<2> row_gws(uiHeight, (iRadiusAligned + uiNumOutputPix + iRadius) *
                                      DivUp(uiWidth, uiNumOutputPix));
    sycl::range<2> row_lws(1, iRadiusAligned + uiNumOutputPix + iRadius);

    // Set global and local work sizes for column kernel
    sycl::range<1> col_gws(64 * DivUp(uiWidth, 64));
    sycl::range<1> col_lws(64);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iCycles; i++) {
      // Launch row kernel
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<sycl::uchar4, 1> uc4LocalData(
           sycl::range<1>(iRadiusAligned + uiNumOutputPix + iRadius), cgh);
        cgh.parallel_for<class row_kernel>(
          sycl::nd_range<2>(row_gws, row_lws), [=] (sycl::nd_item<2> item) {
          int lid = item.get_local_id(1);
          int gidx = item.get_group(1);
          int gidy = item.get_group(0);

          int globalPosX = gidx * uiNumOutputPix + lid - iRadiusAligned;
          int globalPosY = gidy;
          int iGlobalOffset = globalPosY * uiWidth + globalPosX;

          // Read global data into LMEM
          if (globalPosX >= 0 && globalPosX < uiWidth)
          {
              uc4LocalData[lid] = cmBufIn[iGlobalOffset];
          }
          else
              uc4LocalData[lid] = {0, 0, 0, 0};

          item.barrier(sycl::access::fence_space::local_space);

          if((globalPosX >= 0) && (globalPosX < uiWidth) && (lid >= iRadiusAligned) &&
             (lid < (iRadiusAligned + (int)uiNumOutputPix)))
          {
              // Init summation registers to zero
              sycl::float4 f4Sum = {0.0f, 0.0f, 0.0f, 0.0f};

              // Do summation, using inline function to break up uint value from LMEM into independent RGBA values
              int iOffsetX = lid - iRadius;
              int iLimit = iOffsetX + (2 * iRadius) + 1;
              for(; iOffsetX < iLimit; iOffsetX++)
              {
                  f4Sum.x() += uc4LocalData[iOffsetX].x();
                  f4Sum.y() += uc4LocalData[iOffsetX].y();
                  f4Sum.z() += uc4LocalData[iOffsetX].z();
                  f4Sum.w() += uc4LocalData[iOffsetX].w();
              }

              // Use inline function to scale and convert registers to packed RGBA values in a sycl::uchar4,
              // and write back out to GMEM
              cmBufTmp[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
          }
        });
      });

      // Launch column kernel
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class col_kernel>(
          sycl::nd_range<1>(col_gws, col_lws), [=] (sycl::nd_item<1> item) {
          int globalPosX = item.get_global_id(0);
          if (globalPosX >= uiWidth) return;
          auto uiInputImage = cmBufTmp + globalPosX;
          auto uiOutputImage = cmBufOut + globalPosX;

          sycl::float4 top_color = rgbaUintToFloat4(uiInputImage[0]);
          sycl::float4 bot_color = rgbaUintToFloat4(uiInputImage[(uiHeight - 1) * uiWidth]);

          sycl::float4 radius = {iRadius, iRadius, iRadius, iRadius};
          sycl::float4 f4Sum = top_color * radius ;
          for (int y = 0; y < iRadius + 1; y++)
          {
              f4Sum += rgbaUintToFloat4(uiInputImage[y * uiWidth]);
          }
          uiOutputImage[0] = rgbaFloat4ToUint(f4Sum, fScale);
          for(int y = 1; y < iRadius + 1; y++)
          {
              f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
              f4Sum -= top_color;
              uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
          }

          for(int y = iRadius + 1; y < uiHeight - iRadius; y++)
          {
              f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
              f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
              uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
          }

          for (int y = uiHeight - iRadius; y < uiHeight; y++)
          {
              f4Sum += bot_color;
              f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
              uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
          }
        });
     });
   }

   q.wait();
   auto end = std::chrono::steady_clock::now();
   auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
   printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / iCycles);
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage %s <PPM image> <repeat>\n", argv[0]);
    return 1;
  }
  const unsigned int uiMaxImageWidth = 1920;   // Image width
  const unsigned int uiMaxImageHeight = 1080;  // Image height
  unsigned int uiImageWidth = 0;     // Image width
  unsigned int uiImageHeight = 0;    // Image height
  unsigned int* uiInput = NULL;      // Host buffer to hold input image data
  unsigned int* uiTmp = NULL;        // Host buffer to hold intermediate image data
  unsigned int* uiDevOutput = NULL;
  unsigned int* uiHostOutput = NULL;

  bool status = shrLoadPPM4ub(argv[1], (unsigned char**)&uiInput,
                              &uiImageWidth, &uiImageHeight);
  printf("Image Width = %u, Height = %u, bpp = %u, Mask Radius = %u\n",
      uiImageWidth, uiImageHeight, unsigned(sizeof(unsigned int) * 8), RADIUS);
  if (uiImageWidth > uiMaxImageWidth || uiImageHeight > uiMaxImageHeight) {
    printf("Error: Image Dimensions exceed the maximum values");
    status = 0;
  }
  if (!status) {
     free(uiInput);
     return 1;
  }

  printf("Using Local Memory for Row Processing\n\n");

  size_t szBuff = uiImageWidth * uiImageHeight;
  size_t szBuffBytes = szBuff * sizeof (unsigned int);

  // Allocate intermediate and output host image buffers
  uiTmp = (unsigned int*)calloc(szBuff, sizeof(unsigned int));
  uiHostOutput = (unsigned int*)calloc(szBuff, sizeof(unsigned int));
  uiDevOutput = (unsigned int*)malloc(szBuffBytes);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  sycl::uchar4 *cmDevBufIn = sycl::malloc_device<sycl::uchar4>(szBuff, q);
  unsigned int *cmDevBufTmp = sycl::malloc_device<unsigned int>(szBuff, q);
  unsigned int *cmDevBufOut = sycl::malloc_device<unsigned int>(szBuff, q);

  // Copy input data from host to device
  q.memcpy(cmDevBufIn, (sycl::uchar4*)uiInput, sizeof(sycl::uchar4) * szBuff);
  // Initialize buffer data
  q.memset(cmDevBufTmp, 0, szBuffBytes);
  q.memset(cmDevBufOut, 0, szBuffBytes);

  const int iCycles = atoi(argv[2]);

  printf("Warmup..\n");
  BoxFilterGPU (q, cmDevBufIn, cmDevBufTmp, cmDevBufOut,
                uiImageWidth, uiImageHeight, RADIUS, SCALE, iCycles);

  printf("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);

  BoxFilterGPU (q, cmDevBufIn, cmDevBufTmp, cmDevBufOut,
                uiImageWidth, uiImageHeight, RADIUS, SCALE, iCycles);

  // Copy output from device to host
  q.memcpy(uiDevOutput, cmDevBufOut, szBuffBytes).wait();

  // Do filtering on the host
  BoxFilterHost(uiInput, uiTmp, uiHostOutput, uiImageWidth, uiImageHeight, RADIUS, SCALE);

  sycl::free(cmDevBufIn, q);
  sycl::free(cmDevBufTmp, q);
  sycl::free(cmDevBufOut, q);

  // Verification
  int error = 0;
  const int64_t start = RADIUS * uiImageWidth;
  const int64_t end  = (int64_t)szBuff - (int64_t)start;
  for (int64_t i = start; i < end; i++) {
    if (uiDevOutput[i] != uiHostOutput[i]) {
      printf("%ld %08x %08x\n", i, uiDevOutput[i], uiHostOutput[i]);
      error = 1;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  free(uiInput);
  free(uiTmp);
  free(uiDevOutput);
  free(uiHostOutput);
  return 0;
}
