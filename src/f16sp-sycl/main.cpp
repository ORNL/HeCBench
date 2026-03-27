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
#include <cmath>
#include <chrono>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "kernels.h"

void generateInput(sycl::half2 *a, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    sycl::half2 temp;
    temp.x() = (sycl::half)(sqrt(32752.0/size));
    temp.y() = (sycl::half)(sqrt(32752.0/size));
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
  const size_t size_bytes = size * sizeof(sycl::half2);
  const size_t result_bytes = sizeof(float);
  const size_t result2_bytes = sizeof(sycl::half);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  sycl::half2 *a, *b;
  sycl::half2 *d_a, *d_b;

  float r, *d_r;
  sycl::half r2, *d_r2;

  a = (sycl::half2 *) malloc (size_bytes);
  b = (sycl::half2 *) malloc (size_bytes);

  d_a = sycl::malloc_device<sycl::half2>(size, q);
  d_b = sycl::malloc_device<sycl::half2>(size, q);
  d_r = sycl::malloc_device<float>(1, q);
  d_r2 = sycl::malloc_device<sycl::half>(1, q);

  srand(123);
  generateInput(a, size);
  q.memcpy(d_a, a, size_bytes);

  generateInput(b, size);
  q.memcpy(d_b, b, size_bytes);

  printf("\nNumber of elements in the vectors is %zu\n", size * 2);

  // evaluate the grid sizes for performance optimization
  for (int grid = NUM_OF_BLOCKS; grid >= NUM_OF_BLOCKS / 16; grid = grid / 2) {

    printf("\nGPU grid size is %d\n", grid);

    auto fp16Fn = [&](sycl::handler &cgh) {
        sycl::local_accessor<sycl::half2> sm(sycl::range<1>(NUM_OF_THREADS), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid*NUM_OF_THREADS),
                            sycl::range<1>(NUM_OF_THREADS)), [=](sycl::nd_item<1> item) {
          scalarProductKernel_native(d_a, d_b, d_r,
                                     sm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                     size, item);
        });
    };
    auto fp32Fn = [&](sycl::handler &cgh) {
        sycl::local_accessor<sycl::float2> sm(sycl::range<1>(NUM_OF_THREADS), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid*NUM_OF_THREADS),
                            sycl::range<1>(NUM_OF_THREADS)), [=](sycl::nd_item<1> item) {
          scalarProductKernel_native_fp32(d_a, d_b, d_r,
                                          sm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                          size, item);
        });
    };

    // warmup
    for (int i = 0; i < 1000; i++) {
      q.submit(fp16Fn);
    }
    q.wait();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.memset(d_r, 0, result_bytes);
      q.submit(fp16Fn);
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

    q.memcpy(&r, d_r, result_bytes).wait();
    printf("Error rate: %e\n", fabsf(r - 65504.f)/65504.f);

    // warmup
    for (int i = 0; i < 1000; i++) {
      q.submit(fp32Fn);
    }
    q.wait();

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.memset(d_r, 0, result_bytes);
      q.submit(fp32Fn);
    }

    q.wait();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

    q.memcpy(&r, d_r, result_bytes).wait();
    printf("Error rate: %e\n", fabsf(r - 65504.f)/65504.f);

    // warmup
    for (int i = 0; i < 1000; i++) {
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(grid*NUM_OF_THREADS),
                            sycl::range<1>(NUM_OF_THREADS)),
          [=](sycl::nd_item<1> item) {
          scalarProductKernel_native2_fp32(d_a, d_b, d_r, size, item);
        });
      });
    }
    q.wait();

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.memset(d_r, 0, result_bytes);
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(grid*NUM_OF_THREADS),
                            sycl::range<1>(NUM_OF_THREADS)),
          [=](sycl::nd_item<1> item) {
          scalarProductKernel_native2_fp32(d_a, d_b, d_r, size, item);
        });
      });
    }

    q.wait();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

    q.memcpy(&r, d_r, result_bytes).wait();
    printf("Error rate: %e\n", fabsf(r - 65504.f)/65504.f);
  }

  printf("\n");
  // library-based dot product
  for (int i = 0; i < 1000; i++) {
    oneapi::mkl::blas::dot(q, size*2, (sycl::half*)d_a, 1, (sycl::half*)d_b, 1, d_r2);
  }
  q.wait();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    oneapi::mkl::blas::dot(q, size*2, (sycl::half*)d_a, 1, (sycl::half*)d_b, 1, d_r2);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (mkl::blas::dot) execution time %f (us)\n", (time * 1e-3f) / repeat);
  q.memcpy(&r2, d_r2, result2_bytes).wait();
  printf("Error rate: %e\n", fabsf((float)r2 - 65504.f)/65504.f);

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_r, q);
  sycl::free(d_r2, q);
  free(a);
  free(b);

  return EXIT_SUCCESS;
}
