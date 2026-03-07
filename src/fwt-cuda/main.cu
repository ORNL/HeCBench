/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * Walsh transforms belong to a class of generalized Fourier transformations.
 * They have applications in various fields of electrical engineering
 * and numeric theory. In this sample we demonstrate efficient implementation
 * of naturally-ordered Walsh transform
 * (also known as Walsh-Hadamard or Hadamard transform) in CUDA and its
 * particular application to dyadic convolution computation.
 * Refer to excellent Jorg Arndt's "Algorithms for Programmers" textbook
 * http://www.jjj.de/fxt/fxtbook.pdf (Chapter 22)
 *
 * Victor Podlozhnyuk (vpodlozhnyuk@nvidia.com)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

// Reference CPU FWT
extern"C" void fwtCPU(float *h_Output, float *h_Input, int log2N);
extern"C" void slowWTcpu(float *h_Output, float *h_Input, int log2N);
extern "C" void dyadicConvolutionCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int log2dataN,
    int log2kernelN);

// GPU FWT
#include "kernels.cu"

// Data configuration
const int log2Data = 23;
const int dataN = 1 << log2Data;
const int DATA_SIZE = dataN * sizeof(float);

const int log2Kernel = 7;
const int kernelN = 1 << log2Kernel;
const int KERNEL_SIZE = kernelN * sizeof(float);

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  double delta, ref, sum_delta2, sum_ref2, L2norm;

  int i;

  printf("Data length: %i; kernel length: %i\n", dataN, kernelN);

  printf("Initializing data...\n");
  float *h_Kernel    = (float *)malloc(KERNEL_SIZE);
  float *h_Data      = (float *)malloc(DATA_SIZE);
  float *h_ResultCPU = (float *)malloc(DATA_SIZE);
  float *h_ResultGPU = (float *)malloc(DATA_SIZE);

  srand(123);
  for (i = 0; i < kernelN; i++)
  {
    h_Kernel[i] = (float)rand() / (float)RAND_MAX;
  }

  for (i = 0; i < dataN; i++)
  {
    h_Data[i] = (float)rand() / (float)RAND_MAX;
  }

  printf("Running GPU dyadic convolution using Fast Walsh Transform...\n");

  float *d_Data, *d_Kernel;
  cudaMalloc((void **)&d_Kernel, DATA_SIZE);
  cudaMalloc((void **)&d_Data, DATA_SIZE);

  float total_time = 0.f;

  for (i = 0; i < repeat; i++)
  {
    cudaMemset(d_Kernel, 0, DATA_SIZE);
    cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Data, h_Data, DATA_SIZE, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    fwtBatchGPU(d_Data, 1, log2Data);
    fwtBatchGPU(d_Kernel, 1, log2Data);
    modulateGPU(d_Data, d_Kernel, dataN);
    fwtBatchGPU(d_Data, 1, log2Data);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }
  printf("Average device execution time %f (s)\n", (total_time * 1e-9f) / repeat);

  printf("Reading back GPU results...\n");
  cudaMemcpy(h_ResultGPU, d_Data, DATA_SIZE, cudaMemcpyDeviceToHost);

  cudaFree(d_Data);
  cudaFree(d_Kernel);

  printf("Running straightforward CPU dyadic convolution...\n");
  dyadicConvolutionCPU(h_ResultCPU, h_Data, h_Kernel, log2Data, log2Kernel);

  printf("Comparing the results...\n");
  sum_delta2 = 0;
  sum_ref2   = 0;

  for (i = 0; i < dataN; i++)
  {
    delta       = h_ResultCPU[i] - h_ResultGPU[i];
    ref         = h_ResultCPU[i];
    sum_delta2 += delta * delta;
    sum_ref2   += ref * ref;
  }

  L2norm = sqrt(sum_delta2 / sum_ref2);

  printf("Shutting down...\n");
  free(h_ResultGPU);
  free(h_ResultCPU);
  free(h_Data);
  free(h_Kernel);

  printf("L2 norm: %E\n", L2norm);
  printf(L2norm < 1e-6 ? "PASS\n" : "FAIL\n");
  if (L2norm >= 1e-6) exit(1);
}
