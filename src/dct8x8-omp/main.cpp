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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "DCT8x8.h"

void DCT8x8(
    float* d_Dst,
    const float* d_Src,
    unsigned int stride,
    unsigned int imageH,
    unsigned int imageW,
    int dir
);

void Verify(const float* h_OutputGPU, 
                  float* h_OutputCPU, 
            const float* h_Input, 
            const unsigned int stride,
            const unsigned int imageH,
            const unsigned int imageW,
            const int dir )
{
  printf("Comparing against Host/C++ computation...\n"); 
  DCT8x8CPU(h_OutputCPU, h_Input, stride, imageH, imageW, dir);
  double sum = 0, delta = 0;
  double L2norm;
  for(unsigned int i = 0; i < imageH; i++)
    for(unsigned int j = 0; j < imageW; j++){
      sum += h_OutputCPU[i * stride + j] * h_OutputCPU[i * stride + j];
      delta += (h_OutputGPU[i * stride + j] - h_OutputCPU[i * stride + j]) * 
               (h_OutputGPU[i * stride + j] - h_OutputCPU[i * stride + j]);
    }
  L2norm = sqrt(delta / sum);
  printf("Relative L2 norm: %.3e\n\n", L2norm);
  if (L2norm < 1E-6) 
    printf("PASS\n"); 
  else
    printf("FAIL\n");
  if (L2norm >= 1E-6) exit(1);
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <image width> <image height> <repeat>\n", argv[0]);
    return 1;
  }
  const unsigned int imageW = atoi(argv[1]);
  const unsigned int imageH = atoi(argv[2]);
  const int numIterations = atoi(argv[3]);
  const unsigned int stride = imageW;

  printf("Allocating and initializing host memory...\n");
  float *h_Input, *h_OutputCPU, *h_OutputGPU;
  h_Input     = (float *)malloc(imageH * stride * sizeof(float));
  h_OutputCPU = (float *)malloc(imageH * stride * sizeof(float));
  h_OutputGPU = (float *)malloc(imageH * stride * sizeof(float));

  srand(2009);
  for(unsigned int i = 0; i < imageH; i++)
    for(unsigned int j = 0; j < imageW; j++)
      h_Input[i * stride + j] = (float)rand() / (float)RAND_MAX;

#pragma omp target data map(to: h_Input[0:imageH * stride]) \
                        map(alloc: h_OutputGPU[0:imageH * stride]) 
{
  printf("Performing Forward DCT8x8 of %u x %u image on the device\n\n", imageH, imageW);

  int dir = DCT_FORWARD;

  auto start = std::chrono::steady_clock::now();

  for(int iter = 0; iter < numIterations; iter++)
    DCT8x8(
        h_OutputGPU,
        h_Input,
        stride,
        imageH,
        imageW,
        dir );

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average DCT8x8 kernel execution time %f (s)\n", (time * 1e-9f) / numIterations);

  #pragma omp target update from(h_OutputGPU[0:imageH * stride])

  Verify(h_OutputGPU, h_OutputCPU, h_Input, stride, imageH, imageW, dir);

  printf("Performing Inverse DCT8x8 of %u x %u image on the device\n\n", imageH, imageW);

  dir = DCT_INVERSE;

  start = std::chrono::steady_clock::now();

  for(int iter = 0; iter < numIterations; iter++)
    DCT8x8(
        h_OutputGPU,
        h_Input,
        stride,
        imageH,
        imageW,
        dir );

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average IDCT8x8 kernel execution time %f (s)\n", (time * 1e-9f) / numIterations);

  #pragma omp target update from(h_OutputGPU[0:imageH * stride])

  Verify(h_OutputGPU, h_OutputCPU, h_Input, stride, imageH, imageW, dir);

}

  free(h_OutputGPU);
  free(h_OutputCPU);
  free(h_Input);

  return 0;
}
