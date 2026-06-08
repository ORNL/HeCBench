/**
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cublas_v2.h>
#include "kernels.h"

const int WARMUP = 1000;

struct Half2Mul {
  const half2* a;
  const half2* b;
  Half2Mul(const half2* _a, const half2* _b) : a(_a), b(_b) {}

  __host__ __device__
  float operator()(int idx) const {
    half2 prod = __hmul2(a[idx], b[idx]);
    return __low2float(prod) + __high2float(prod);
  }
};

// The analytical result of dot product is 65504
void generateInput(half2 * a, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    half2 temp;
    temp.x = (half)(sqrt(32752.0/size));
    temp.y = (half)(sqrt(32752.0/size));
    a[i] = temp;
  }
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const size_t size = NUM_OF_BLOCKS*NUM_OF_THREADS;
  const size_t size_bytes = size * sizeof(half2);
  const size_t result_bytes = sizeof(float);
  const size_t result2_bytes = sizeof(half);

  half2 *a, *b;
  half2 *d_a, *d_b;

  float r, *d_r;
  half r2, *d_r2;

  a = (half2*) malloc (size_bytes);
  b = (half2*) malloc (size_bytes);
  cudaMalloc((void**)&d_a, size_bytes);
  cudaMalloc((void**)&d_b, size_bytes);

  cudaMalloc((void**)&d_r, result_bytes);
  cudaMalloc((void**)&d_r2, result2_bytes);

  srand(123);
  generateInput(a, size);
  cudaMemcpy(d_a, a, size_bytes, cudaMemcpyHostToDevice);

  generateInput(b, size);
  cudaMemcpy(d_b, b, size_bytes, cudaMemcpyHostToDevice);

  printf("\nNumber of elements in the vectors is %zu\n", size * 2);

  // evaluate the impact of grid sizes on performance
  for (int grid = NUM_OF_BLOCKS; grid >= NUM_OF_BLOCKS / 16; grid = grid / 2) {

    printf("\nGPU grid size is %d\n", grid);

    // WARMUP
    for (int i = 0; i < WARMUP; i++)
      scalarProductKernel_intrinsics<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      cudaMemset(d_r, 0, result_bytes);
      scalarProductKernel_intrinsics<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel1 execution time %f (us)\n", (time * 1e-3f) / repeat);

    cudaMemcpy(&r, d_r, result_bytes, cudaMemcpyDeviceToHost);
    printf("Error rate: %e\n", fabsf(r - 65504.f)/65504.f);

    // WARMUP
    for (int i = 0; i < WARMUP; i++)
      scalarProductKernel_native_fp32<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

    cudaDeviceSynchronize();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      cudaMemset(d_r, 0, result_bytes);
      scalarProductKernel_native_fp32<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
    }

    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel2 execution time %f (us)\n", (time * 1e-3f) / repeat);

    cudaMemcpy(&r, d_r, result_bytes, cudaMemcpyDeviceToHost);
    printf("Error rate: %e\n", fabsf(r - 65504.f)/65504.f);

    // WARMUP
    for (int i = 0; i < WARMUP; i++)
      scalarProductKernel_native2_fp32<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

    cudaDeviceSynchronize();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      cudaMemset(d_r, 0, result_bytes);
      scalarProductKernel_native2_fp32<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
    }

    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel3 execution time %f (us)\n", (time * 1e-3f) / repeat);

    cudaMemcpy(&r, d_r, result_bytes, cudaMemcpyDeviceToHost);
    printf("Error rate: %e\n", fabsf(r - 65504.f)/65504.f);
  }

  printf("\n");
  // library-based dot product
  cublasHandle_t h;
  cublasCreate(&h);
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);

  cudaDataType xType, yType, rType, eType;
  xType = yType = rType = CUDA_R_16F;
  eType = CUDA_R_32F;

  // WARMUP
  for (int i = 0; i < WARMUP; i++) {
    cublasDotEx_64(h, size*2, (half*)d_a, xType, 1, (half*)d_b,
                yType, 1, d_r2, rType, eType);
  }
  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    cublasDotEx_64(h, size*2, (half*)d_a, xType, 1, (half*)d_b,
                yType, 1, d_r2, rType, eType);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average cublasDotEx_64 execution time %f (us)\n", (time * 1e-3f) / repeat);
  cudaMemcpy(&r2, d_r2, result2_bytes, cudaMemcpyDeviceToHost);
  printf("Error rate: %e\n", fabsf((float)r2 - 65504.f)/65504.f);

  printf("\n");

  for (int i = 0; i < WARMUP + repeat; i++) {
    if (i == WARMUP) {
      cudaDeviceSynchronize();
      start = std::chrono::steady_clock::now();
    }

    thrust::counting_iterator<size_t> counting(0);
    auto iter = thrust::make_transform_iterator(counting, Half2Mul(d_a, d_b));

    void* d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, iter, d_r, size);
    if (temp_bytes != 0) cudaMalloc(&d_temp, temp_bytes);
    auto status = cub::DeviceReduce::Sum(d_temp, temp_bytes, iter, d_r, size);
    if (status != cudaSuccess) printf("cub::DeviceReduce::Sum failed\n");
    if (temp_bytes != 0) cudaFree(d_temp);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average cub::DeviceReduce::Sum execution time: %f (us)\n", time * 1e-3f / repeat);

  cudaMemcpy(&r, d_r, result_bytes, cudaMemcpyDeviceToHost);
  printf("Error rate: %e\n", fabsf(r - 65504.f)/65504.f);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_r);
  cudaFree(d_r2);
  cublasDestroy(h);
  free(a);
  free(b);

  return EXIT_SUCCESS;
}
