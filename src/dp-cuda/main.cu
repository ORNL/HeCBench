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

// *********************************************************************
// A simple demo application that implements a
// vector dot product computation between two arrays.
//
// Runs computations with on the GPU device and then checks results
// *********************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include <execution>
#include <random>
#include <cuda.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include "shrUtils.h"

inline void GPU_CHECK(cudaError_t status) {
  if (status != cudaSuccess) {
    printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
    throw std::logic_error("cuda API failed");
  }
}

inline void BLAS_CHECK(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS API failed with status %d", status);
    throw std::logic_error("cuBLAS API failed");
  }
}

template <typename T>
__global__
void dot_product(const T *__restrict__ a,
                 const T *__restrict__ b,
                       T *__restrict__ d,
                 const size_t n)
{
  size_t iGID = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  T sum = 0;
  for(size_t idx = iGID; idx < n; idx += gridDim.x * blockDim.x) {
    size_t iInOffset = idx * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
      sum += a[iInOffset + i] * b[iInOffset + i];
  }

  using BlockReduce = cub::BlockReduce<T, 1024>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T aggregate = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0) {
    atomicAdd(d, aggregate);
  }
}

template <typename T>
void dot (const size_t iNumElements, const int iNumIterations)
{
  // set and log Global and Local work size dimensions
  int szLocalWorkSize = 1024;
  // rounded up to the nearest multiple of the LocalWorkSize
  size_t szGlobalWorkSize = shrRoundUp(szLocalWorkSize, iNumElements);

  printf("Global Work Size \t\t= %zu\nLocal Work Size \t\t= %d\n",
         szGlobalWorkSize, szLocalWorkSize);

  const size_t src_size = szGlobalWorkSize;
  const size_t src_size_bytes = src_size * sizeof(T);

  const size_t grid_size = shrRoundUp(1, szGlobalWorkSize / (szLocalWorkSize * 4));

  // Allocate and initialize host arrays
  T* srcA = (T*) malloc (src_size_bytes);
  T* srcB = (T*) malloc (src_size_bytes);
  T  dst, dst_ref = 0;

  size_t i;
  std::mt19937 engine(19937);
  std::uniform_int_distribution<int> dis (-32, 32);

  for (i = 0; i < iNumElements; ++i)
  {
    srcA[i] = dis(engine);
    srcB[i] = dis(engine);
    dst_ref += srcA[i] * srcB[i];
  }
  for (i = iNumElements; i < src_size; ++i) {
    srcA[i] = srcB[i] = 0;
  }

  T *d_srcA, *d_srcB, *d_dst;

  GPU_CHECK(cudaMalloc((void**)&d_srcA, src_size_bytes));
  GPU_CHECK(cudaMemcpy(d_srcA, srcA, src_size_bytes, cudaMemcpyHostToDevice));

  GPU_CHECK(cudaMalloc((void**)&d_srcB, src_size_bytes));
  GPU_CHECK(cudaMemcpy(d_srcB, srcB, src_size_bytes, cudaMemcpyHostToDevice));

  GPU_CHECK(cudaMalloc((void**)&d_dst, sizeof(T)));

  dim3 grid (grid_size);
  dim3 block (szLocalWorkSize);

  // warmup
  for (i = 0; i < 100; i++) {
    GPU_CHECK(cudaMemset(d_dst, 0, sizeof(T)));
    dot_product<<<grid, block>>>(d_srcA, d_srcB, d_dst, src_size / 4);
  }

  GPU_CHECK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    GPU_CHECK(cudaMemset(d_dst, 0, sizeof(T)));
    dot_product<<<grid, block>>>(d_srcA, d_srcB, d_dst, src_size / 4);
  }

  GPU_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);

  GPU_CHECK(cudaMemcpy(&dst, d_dst, sizeof(T), cudaMemcpyDeviceToHost));
  printf("%s\n\n", dst == dst_ref ? "PASS" : "FAIL");
  if (dst != dst_ref) exit(1);

  cublasHandle_t h;
  BLAS_CHECK(cublasCreate(&h));
  BLAS_CHECK(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE));

  // warmup
  for (i = 0; i < 100; i++) {
    cudaDataType xType, yType, rType, eType;
    if constexpr (std::is_same<T, double>::value) {
      xType = yType = rType = eType = CUDA_R_64F;
    } else if constexpr (std::is_same<T, float>::value) {
      xType = yType = rType = eType = CUDA_R_32F;
    }

    BLAS_CHECK(cublasDotEx(h, iNumElements, d_srcA, xType, 1, d_srcB,
                           yType, 1, d_dst, rType, eType));
  }

  GPU_CHECK(cudaDeviceSynchronize());
  start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    cudaDataType xType, yType, rType, eType;
    if constexpr (std::is_same<T, double>::value) {
      xType = yType = rType = eType = CUDA_R_64F;
    } else if constexpr (std::is_same<T, float>::value) {
      xType = yType = rType = eType = CUDA_R_32F;
    }

    BLAS_CHECK(cublasDotEx(h, iNumElements, d_srcA, xType, 1, d_srcB,
                           yType, 1, d_dst, rType, eType));
  }

  GPU_CHECK(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average cublasDotEx execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);

  GPU_CHECK(cudaMemcpy(&dst, d_dst, sizeof(T), cudaMemcpyDeviceToHost));
  printf("%s\n\n", dst == dst_ref ? "PASS" : "FAIL");
  if (dst != dst_ref) exit(1);

  // warmup
  for (i = 0; i < 100; i++) {
    dst = std::transform_reduce(std::execution::par_unseq,
                                d_srcA, d_srcA + iNumElements, d_srcB, T(0));
  }

  start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    dst = std::transform_reduce(std::execution::par_unseq,
                                d_srcA, d_srcA + iNumElements, d_srcB, T(0));
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average std::transform_reduce execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);
  printf("%s\n\n", dst == dst_ref ? "PASS" : "FAIL");
  if (dst != dst_ref) exit(1);

  GPU_CHECK(cudaFree(d_dst));
  GPU_CHECK(cudaFree(d_srcA));
  GPU_CHECK(cudaFree(d_srcB));
  BLAS_CHECK(cublasDestroy(h));

  free(srcA);
  free(srcB);
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t iNumElements = atol(argv[1]);
  const int iNumIterations = atoi(argv[2]);

  printf("------------- Data type is Float32 ---------------\n");
  dot<float>(iNumElements, iNumIterations);
  printf("------------- Data type is Float64 ---------------\n");
  dot<double>(iNumElements, iNumIterations);

  return EXIT_SUCCESS;
}
