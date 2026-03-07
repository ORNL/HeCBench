/*
 * Copyright 2010 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */


#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include "kernels.h"

#define NUM_BLOCKS 1024
#define BLOCK_SIZE 256

template <typename T>
void testMin (T *h_ptr, T *d_ptr, const int repeat, const char* name) {
  hipMemcpy(d_ptr, h_ptr, sizeof(T), hipMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(atomicMinDerived<T>), NUM_BLOCKS, BLOCK_SIZE, 0, 0, d_ptr);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic min for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  hipMemcpy(h_ptr, d_ptr, sizeof(T), hipMemcpyDeviceToHost);
}

template <typename T>
void testMax (T *h_ptr, T *d_ptr, const int repeat, const char* name) {
  hipMemcpy(d_ptr, h_ptr, sizeof(T), hipMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(atomicMaxDerived<T>), NUM_BLOCKS, BLOCK_SIZE, 0, 0, d_ptr);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic max for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  hipMemcpy(h_ptr, d_ptr, sizeof(T), hipMemcpyDeviceToHost);
}

template <typename T>
void testAdd (T *h_ptr, T *d_ptr, const int repeat, const char* name) {
  hipMemcpy(d_ptr, h_ptr, sizeof(T), hipMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(atomicAddDerived<T>), NUM_BLOCKS, BLOCK_SIZE, 0, 0, d_ptr);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic add for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  hipMemcpy(h_ptr, d_ptr, sizeof(T), hipMemcpyDeviceToHost);
}

int main(int argc, char** argv) {

  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned long long res_u64[3] = {ULONG_MAX,0,0};
  long long res_s64[3] = {LONG_MAX,LONG_MIN,0};
  double res_f64[3] = {DBL_MAX,DBL_MIN,0};

  unsigned long long *d_res_u64;   
  long long *d_res_s64;   
  double *d_res_f64;   

  hipMalloc((void**)&d_res_u64, 3*sizeof(unsigned long long));
  hipMalloc((void**)&d_res_s64, 3*sizeof(long long));
  hipMalloc((void**)&d_res_f64, 3*sizeof(double));

  testMin<unsigned long long>(res_u64, d_res_u64, repeat, "U64");
  testMin<long long>(res_s64, d_res_s64, repeat, "S64");
  testMin<double>(res_f64, d_res_f64, repeat, "F64");

  testMax<unsigned long long>(res_u64+1, d_res_u64+1, repeat, "U64");
  testMax<long long>(res_s64+1, d_res_s64+1, repeat, "S64");
  testMax<double>(res_f64+1, d_res_f64+1, repeat, "F64");

  // the add kernels are slow
  testAdd<unsigned long long>(res_u64+2, d_res_u64+2, 1, "U64");
  testAdd<long long>(res_s64+2, d_res_s64+2, 1, "S64");
  testAdd<double>(res_f64+2, d_res_f64+2, 1, "F64");

  unsigned long long bound = NUM_BLOCKS*BLOCK_SIZE;
  unsigned long long sum = 0; 
  for (unsigned int i = 1; i <= bound; i++) sum += i;

  bool error = false;
  if (res_u64[0] != 1ULL || res_s64[0] != 1LL || res_f64[0] != 1.0) {
    error = true;
    printf("atomic min results: %llu %lld %lf\n", res_u64[0], res_s64[0], res_f64[0]);
  }
  if (res_u64[1] != bound || res_s64[1] != (long long)bound || res_f64[1] != (double)bound) {
    error = true;
    printf("atomic max results: %llu %lld %lf\n", res_u64[1], res_s64[1], res_f64[1]);
  }
  if (res_u64[2] != sum || res_s64[2] != (long long)sum || res_f64[2] != (double)sum) {
    error = true;
    printf("atomic add results: %llu %lld %lf\n", res_u64[2], res_s64[2], res_f64[2]);
  }
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  hipFree(d_res_u64);
  hipFree(d_res_s64);
  hipFree(d_res_f64);
  return 0;
}
