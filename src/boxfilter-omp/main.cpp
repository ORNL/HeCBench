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
#include <omp.h>
#include "shrUtils.h"
#include <cstdlib>

typedef struct __attribute__((__aligned__(4)))
{
  unsigned char x;
  unsigned char y;
  unsigned char z;
  unsigned char w;
} uchar4;

typedef struct __attribute__((__aligned__(16)))
{
  float x;
  float y;
  float z;
  float w;
} float4;

extern
void BoxFilterHost( unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage,
                    int uiWidth, int uiHeight, int iRadius, float fScale );


const unsigned int RADIUS = 10;                    // initial radius of 2D box filter mask
const float SCALE = 1.0f/(2.0f * RADIUS + 1.0f);  // precalculated GV rescaling value

inline uint DivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#pragma omp declare target
// Helper function to convert float[4] rgba color to 32-bit unsigned integer
//*****************************************************************
float4 rgbaUintToFloat4(unsigned int c)
{
    float4 rgba;
    rgba.x = c & 0xff;
    rgba.y = (c >> 8) & 0xff;
    rgba.z = (c >> 16) & 0xff;
    rgba.w = (c >> 24) & 0xff;
    return rgba;
}

uchar4 rgbaUintToUchar4(unsigned int c)
{
    uchar4 rgba;
    rgba.x = c & 0xff;
    rgba.y = (c >> 8) & 0xff;
    rgba.z = (c >> 16) & 0xff;
    rgba.w = (c >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
unsigned int rgbaFloat4ToUint(float4 rgba, float fScale)
{
    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x * fScale);
    uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y * fScale)) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z * fScale)) << 16);
    uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w * fScale)) << 24);
    return uiPackedPix;
}

inline float4 operator*(float4 a, float4 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w};
}

inline void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
#pragma omp end declare target

void BoxFilterGPU ( unsigned int *uiInput,
                    unsigned int *uiTmp,
                    unsigned int *uiDevOutput,
                    const int uiWidth,
                    const int uiHeight,
                    const int iRadius,
                    const float fScale,
                    const float iCycles )
{
  const int szMaxWorkgroupSize = 256;
  const int iRadiusAligned = ((iRadius + 15)/16) * 16;  // 16
  unsigned int uiNumOutputPix = 64;  // Default output pix per workgroup

  if (szMaxWorkgroupSize < (iRadiusAligned + uiNumOutputPix + iRadius))
    uiNumOutputPix = szMaxWorkgroupSize - iRadiusAligned - iRadius;

  // Set team and thread sizes for row kernel // Workgroup padded left and right
  const int uiBlockWidth = DivUp(uiWidth, uiNumOutputPix);
  const int numTeams = uiHeight * uiBlockWidth;
  const int blockSize = iRadiusAligned + uiNumOutputPix + iRadius;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iCycles; i++) {
    // Launch row kernel
    #pragma omp target teams num_teams(numTeams) thread_limit(blockSize)
    {
      uchar4 uc4LocalData[90]; //16+64+10;
      #pragma omp parallel
      {
        int lid = omp_get_thread_num();
        int gidx = omp_get_team_num() % uiBlockWidth;
        int gidy = omp_get_team_num() / uiBlockWidth;

        int globalPosX = gidx * uiNumOutputPix + lid - iRadiusAligned;
        int globalPosY = gidy;
        int iGlobalOffset = globalPosY * uiWidth + globalPosX;

        // Read global data into LMEM
        if (globalPosX >= 0 && globalPosX < uiWidth)
            // IBM xlc: no known conversion from 'unsigned int *' to 'uchar4 *'
            uc4LocalData[lid] = rgbaUintToUchar4(uiInput[iGlobalOffset]);
        else
            uc4LocalData[lid] = {0, 0, 0, 0};

        #pragma omp barrier

        if((globalPosX >= 0) && (globalPosX < uiWidth) && (lid >= iRadiusAligned) &&
           (lid < (iRadiusAligned + (int)uiNumOutputPix)))
        {
            // Init summation registers to zero
            float4 f4Sum = {0.0f, 0.0f, 0.0f, 0.0f};

            // Do summation, using inline function to break up uint value from LMEM into independent RGBA values
            int iOffsetX = lid - iRadius;
            int iLimit = iOffsetX + (2 * iRadius) + 1;
            for(; iOffsetX < iLimit; iOffsetX++)
            {
                f4Sum.x += uc4LocalData[iOffsetX].x;
                f4Sum.y += uc4LocalData[iOffsetX].y;
                f4Sum.z += uc4LocalData[iOffsetX].z;
                f4Sum.w += uc4LocalData[iOffsetX].w;
            }

            // Use inline function to scale and convert registers to packed RGBA values in a uchar4,
            // and write back out to GMEM
            uiTmp[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
        }
      }
    }

    // Launch column kernel
    #pragma omp target teams distribute parallel for thread_limit(64)
    for (size_t globalPosX = 0; globalPosX < uiWidth; globalPosX++) {
      unsigned int* uiInputImage = &uiTmp[globalPosX];
      unsigned int* uiOutputImage = &uiDevOutput[globalPosX];

      float4 f4Sum;
      float4 f4iRadius = {(float)iRadius, (float)iRadius, (float)iRadius, (float)iRadius};
      float4 top_color = rgbaUintToFloat4(uiInputImage[0]);
      float4 bot_color = rgbaUintToFloat4(uiInputImage[(uiHeight - 1) * uiWidth]);

      f4Sum = top_color * f4iRadius;
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
    }
  }
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

  // Allocate intermediate and output host image buffers
  uiTmp = (unsigned int*)calloc(szBuff, sizeof(unsigned int));
  uiHostOutput = (unsigned int*)calloc(szBuff, sizeof(unsigned int));
  uiDevOutput = (unsigned int*)calloc(szBuff, sizeof(unsigned int));

  #pragma omp target data map(to: uiInput[0:szBuff]) \
                          map(to: uiTmp[0:szBuff]) \
                          map(tofrom: uiDevOutput[0:szBuff])
  {
    const int iCycles = atoi(argv[2]);

    printf("Warmup..\n");
    BoxFilterGPU (uiInput, uiTmp, uiDevOutput,
                  uiImageWidth, uiImageHeight, RADIUS, SCALE, iCycles);


    printf("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);
    BoxFilterGPU (uiInput, uiTmp, uiDevOutput,
                  uiImageWidth, uiImageHeight, RADIUS, SCALE, iCycles);
  }

  // Do filtering on the host
  BoxFilterHost(uiInput, uiTmp, uiHostOutput, uiImageWidth, uiImageHeight, RADIUS, SCALE);

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

  free(uiInput);
  free(uiTmp);
  free(uiDevOutput);
  free(uiHostOutput);
  if (error) return 1;
  return 0;
}
