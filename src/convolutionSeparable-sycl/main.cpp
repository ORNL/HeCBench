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
#include <sycl/sycl.hpp>
#include "conv.h"

int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <image width> <image height> <repeat>\n", argv[0]); 
    return 1;
  }
  const unsigned int imageW = atoi(argv[1]);
  const unsigned int imageH = atoi(argv[2]);
  const int numIterations = atoi(argv[3]);

  const size_t kernelSize_bytes = KERNEL_LENGTH * sizeof(float);
  const size_t imageSize = (size_t)imageW * imageH;
  const size_t imageSize_bytes = imageSize * sizeof(float);

  float* h_Kernel    = (float*)malloc(kernelSize_bytes);
  float* h_Input     = (float*)malloc(imageSize_bytes);
  float* h_Buffer    = (float*)malloc(imageSize_bytes);
  float* h_OutputCPU = (float*)malloc(imageSize_bytes);
  float* h_OutputGPU = (float*)malloc(imageSize_bytes);

  srand(2009);
  for(unsigned int i = 0; i < KERNEL_LENGTH; i++)
    h_Kernel[i] = (float)(rand() % 16);

  for(unsigned int i = 0; i < imageW * imageH; i++)
    h_Input[i] = (float)(rand() % 16);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_Kernel = sycl::malloc_device<float>(KERNEL_LENGTH, q);
  q.memcpy(d_Kernel, h_Kernel, kernelSize_bytes);

  float *d_Input = sycl::malloc_device<float>(imageSize, q);
  q.memcpy(d_Input, h_Input, imageSize_bytes);

  float *d_Buffer = sycl::malloc_device<float>(imageSize, q);
  float *d_Output = sycl::malloc_device<float>(imageSize, q);

  //Just a single run or a warmup iteration
  convolutionRows(
      q,
      d_Buffer,
      d_Input,
      d_Kernel,
      imageW,
      imageH,
      imageW);

  convolutionColumns(
      q,
      d_Output,
      d_Buffer,
      d_Kernel,
      imageW,
      imageH,
      imageW);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for(int iter = 0; iter < numIterations; iter++){
    convolutionRows(
        q,
        d_Buffer,
        d_Input,
        d_Kernel,
        imageW,
        imageH,
        imageW);

    convolutionColumns(
        q,
        d_Output,
        d_Buffer,
        d_Kernel,
        imageW,
        imageH,
        imageW);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / numIterations);

  q.memcpy(h_OutputGPU, d_Output, imageSize_bytes).wait();

  printf("Comparing against Host/C++ computation...\n"); 
  convolutionRowHost(h_Buffer, h_Input, h_Kernel, imageW, imageH, KERNEL_RADIUS);
  convolutionColumnHost(h_OutputCPU, h_Buffer, h_Kernel, imageW, imageH, KERNEL_RADIUS);
  double sum = 0, delta = 0;
  double L2norm;
  for(unsigned int i = 0; i < imageW * imageH; i++){
    delta += (h_OutputCPU[i] - h_OutputGPU[i]) * (h_OutputCPU[i] - h_OutputGPU[i]);
    sum += h_OutputCPU[i] * h_OutputCPU[i];
  }
  L2norm = std::sqrt(delta / sum);
  printf("Relative L2 norm: %.3e\n\n", L2norm);

  free(h_OutputGPU);
  free(h_OutputCPU);
  free(h_Buffer);
  free(h_Input);
  free(h_Kernel);
  sycl::free(d_Buffer, q);
  sycl::free(d_Output, q);
  sycl::free(d_Input, q);
  sycl::free(d_Kernel, q);

  printf("%s\n", L2norm < 1e-6 ? "PASS" : "FAIL");
  if (L2norm >= 1e-6) exit(1);
  return 0;
}
