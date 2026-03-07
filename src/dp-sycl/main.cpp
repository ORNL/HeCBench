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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <random>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "shrUtils.h"

template <typename T>
void dot (const size_t iNumElements, const int iNumIterations) try {
  // set and log Global and Local work size dimensions
  int szLocalWorkSize = 256;
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

  for (i = 0; i < iNumElements ; ++i)
  {
    srcA[i] = dis(engine);
    srcB[i] = dis(engine);
    dst_ref += srcA[i] * srcB[i];
  }
  for (i = iNumElements; i < src_size ; ++i) {
    srcA[i] = srcB[i] = 0;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  T *d_srcA = sycl::malloc_device<T>(src_size, q);
  q.memcpy(d_srcA, srcA, src_size_bytes);

  T *d_srcB = sycl::malloc_device<T>(src_size, q);
  q.memcpy(d_srcB, srcB, src_size_bytes);

  T *d_dst = sycl::malloc_device<T>(1, q);

  sycl::range<1> gws (grid_size * szLocalWorkSize);
  sycl::range<1> lws (szLocalWorkSize);

  auto dot_kernel = [&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      size_t iGID = item.get_global_id(0);
      T sum = 0;
      for(size_t idx = iGID; idx < src_size / 4;
          idx += item.get_local_range(0) * item.get_group_range(0)) {
        size_t iInOffset = idx * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++)
          sum += d_srcA[iInOffset + i] * d_srcB[iInOffset + i];
      }
      T aggregate = sycl::reduce_over_group(item.get_group(), sum, std::plus<T>());
      if (item.get_local_id(0) == 0) {
        sycl::atomic_ref<T, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space> ao (d_dst[0]);
        ao.fetch_add(aggregate);
      }
    });
  };
  // warmup
  for (i = 0; i < 100; i++)
    q.submit(dot_kernel);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    q.memset(d_dst, 0, sizeof(T));
    q.submit(dot_kernel);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);
  q.memcpy(&dst, d_dst, sizeof(T)).wait();
  printf("%s\n\n", dst == dst_ref ? "PASS" : "FAIL");
  if (dst != dst_ref) exit(1);

  // warmup
  for (i = 0; i < 100; i++) {
    oneapi::mkl::blas::dot(q, iNumElements, d_srcA, 1, d_srcB, 1, d_dst);
  }
  q.wait();

  start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    oneapi::mkl::blas::dot(q, iNumElements, d_srcA, 1, d_srcB, 1, d_dst);
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average oneMKL::dot execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);
  q.memcpy(&dst, d_dst, sizeof(T)).wait();
  printf("%s\n\n", dst == dst_ref ? "PASS" : "FAIL");
  if (dst != dst_ref) exit(1);

  // warmup
  for (i = 0; i < 100; i++)
    dst = std::transform_reduce(oneapi::dpl::execution::make_device_policy(q),
                                d_srcA, d_srcA + iNumElements, d_srcB, T(0));

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < iNumIterations; i++) {
    dst = std::transform_reduce(oneapi::dpl::execution::make_device_policy(q),
                                d_srcA, d_srcA + iNumElements, d_srcB, T(0));
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average std::transform_reduce execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);
  printf("%s\n\n", dst == dst_ref ? "PASS" : "FAIL");
  if (dst != dst_ref) exit(1);

  sycl::free(d_dst, q);
  sycl::free(d_srcA, q);
  sycl::free(d_srcB, q);

  free(srcA);
  free(srcB);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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
