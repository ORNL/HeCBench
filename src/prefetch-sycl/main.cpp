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
#include <sycl/sycl.hpp>
#include <cstdlib>

//constexpr int MEM_ADVISE_READ_MOSTLY = PI_MEM_ADVICE_CUDA_SET_READ_MOSTLY;

void add(int n, const float *x, float *y, sycl::nd_item<3> &item)
{
  int index = item.get_global_id(2);
  int stride = item.get_local_range(2) * item.get_group_range(2);
  for (int i = index; i < n; i += stride)
    y[i] += x[i];
}

void prefetch(sycl::queue &q, const int numElements, const int repeat)
{
  printf("Concurrent managed access with prefetch\n");

  float *A = sycl::malloc_shared<float>(numElements, q);
  float *B = sycl::malloc_shared<float>(numElements, q);

  for (int i = 0; i < numElements; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
  }

  q.wait();

  float maxError = 0.0f;

  int blockSize = 256;
  int numBlocks = (numElements + blockSize - 1) / blockSize;
  sycl::range<3> gws (1, 1, numBlocks * blockSize);
  sycl::range<3> lws (1, 1, blockSize);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {

    //q.mem_advise(A, numElements * sizeof(float), MEM_ADVISE_READ_MOSTLY);

    q.prefetch(A, numElements * sizeof(float));
    q.prefetch(B, numElements * sizeof(float));

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k1>(
        sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
        add(numElements, A, B, item);
      });
    }).wait();

    //q.prefetch(B, numElements * sizeof(float)).wait();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (ms)\n", time * 1e-6f / repeat);

  for (int i = 0; i < numElements; i++)
    maxError = fmaxf(maxError, fabsf(B[i]-(repeat+2)));

  sycl::free(A, q);
  sycl::free(B, q);

  bool testResult = (maxError == 0.0f);
  printf("%s\n", testResult ? "PASS" : "FAIL");
}

void naive(sycl::queue &q, const int numElements, const int repeat)
{
  printf("Concurrent managed access without prefetch\n");

  float *A = sycl::malloc_shared<float>(numElements, q);
  float *B = sycl::malloc_shared<float>(numElements, q);

  for (int i = 0; i < numElements; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
  }

  q.wait();

  float maxError = 0.0f;

  int blockSize = 256;
  int numBlocks = (numElements + blockSize - 1) / blockSize;
  sycl::range<3> gws (1, 1, numBlocks * blockSize);
  sycl::range<3> lws (1, 1, blockSize);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k2>(
        sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
        add(numElements, A, B, item);
      });
    }).wait();
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (ms)\n", time * 1e-6f / repeat);

  for (int i = 0; i < numElements; i++)
    maxError = fmaxf(maxError, fabsf(B[i]-(repeat+2)));

  sycl::free(A, q);
  sycl::free(B, q);

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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  bool concurrentManagedAccess = q.get_device().get_info<sycl::info::device::usm_shared_allocations>();

  if(!concurrentManagedAccess) {
    printf("info: concurrent managed access not supported on device. Skipped\n");
    return 0;
  }

  const int numElements = 64 * 1024 * 1024;

  for (int i = 0; i < 10; i++) {
    prefetch(q, numElements, repeat);
  }

  for (int i = 0; i < 10; i++) {
    naive(q, numElements, repeat);
  }
  if (!testResult) return 1;
  return 0;
}
