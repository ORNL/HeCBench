/*
 Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#include <stdio.h>
#include <math.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <cstdlib>

#define HIPCHECK(error)                                                                        \
{                                                                                              \
    hipError_t localError = error;                                                             \
    if (localError != hipSuccess) {                                                            \
        printf("error: %s at %s:%d\n", hipGetErrorString(localError),  __FILE__, __LINE__);    \
    }                                                                                          \
}

__global__
void add(int n, const float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] += x[i];
}

void prefetch (const int gpuDeviceId, const int numElements, const int repeat)
{
  printf("Concurrent managed access with prefetch\n");

  float *A, *B;

  HIPCHECK(hipMallocManaged(&A, numElements*sizeof(float)));
  HIPCHECK(hipMallocManaged(&B, numElements*sizeof(float)));

  for (int i = 0; i < numElements; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
  }

  HIPCHECK(hipDeviceSynchronize());

  float maxError = 0.0f;

  int blockSize = 256;
  int numBlocks = (numElements + blockSize - 1) / blockSize;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(blockSize, 1, 1);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {

    //HIPCHECK(hipMemAdvise(A, numElements*sizeof(float), hipMemAdviseSetReadMostly, hipCpuDeviceId));
    HIPCHECK(hipMemPrefetchAsync(A, numElements*sizeof(float), gpuDeviceId));
    HIPCHECK(hipMemPrefetchAsync(B, numElements*sizeof(float), gpuDeviceId));

    hipLaunchKernelGGL(add, dimGrid, dimBlock, 0, 0, numElements, A, B);

    //HIPCHECK(hipMemPrefetchAsync(B, numElements*sizeof(float), hipCpuDeviceId));
    HIPCHECK(hipDeviceSynchronize());
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (ms)\n", time * 1e-6f / repeat);

  for (int i = 0; i < numElements; i++)
    maxError = fmaxf(maxError, fabsf(B[i]-(repeat+2)));

  HIPCHECK(hipFree(A));
  HIPCHECK(hipFree(B));

  bool testResult = (maxError == 0.0f);
  printf("%s\n", testResult ? "PASS" : "FAIL");
  if (!testResult) exit(1);
}

void naive (const int numElements, const int repeat)
{
  printf("Concurrent managed access without prefetch\n");

  float *A, *B;

  HIPCHECK(hipMallocManaged(&A, numElements*sizeof(float)));
  HIPCHECK(hipMallocManaged(&B, numElements*sizeof(float)));

  for (int i = 0; i < numElements; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
  }

  HIPCHECK(hipDeviceSynchronize());

  float maxError = 0.0f;

  int blockSize = 256;
  int numBlocks = (numElements + blockSize - 1) / blockSize;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(blockSize, 1, 1);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    hipLaunchKernelGGL(add, dimGrid, dimBlock, 0, 0, numElements, A, B);

    HIPCHECK(hipDeviceSynchronize());
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (ms)\n", time * 1e-6f / repeat);

  for (int i = 0; i < numElements; i++)
    maxError = fmaxf(maxError, fabsf(B[i]-(repeat+2)));

  HIPCHECK(hipFree(A));
  HIPCHECK(hipFree(B));

  bool testResult = (maxError == 0.0f);
  printf("%s\n", testResult ? "PASS" : "FAIL");
  if (!testResult) exit(1);
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int p_gpuDevice = 0;
  HIPCHECK(hipSetDevice(p_gpuDevice));
  printf("info: set device to %d\n", p_gpuDevice);

  int concurrentManagedAccess = 0;
  HIPCHECK(hipDeviceGetAttribute(&concurrentManagedAccess,
        hipDeviceAttributeConcurrentManagedAccess,
        p_gpuDevice));
  if(!concurrentManagedAccess) {
    printf("info: concurrent managed access not supported on device %d\n Skipped\n", p_gpuDevice);
    return 0;
  }

  const int numElements = 64 * 1024 * 1024;

  for (int i = 0; i < 10; i++) {
    prefetch(p_gpuDevice, numElements, repeat);
  }

  for (int i = 0; i < 10; i++) {
    naive(numElements, repeat);
  }
  return 0;
}
