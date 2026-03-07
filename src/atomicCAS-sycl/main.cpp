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
#include <sycl/sycl.hpp>

#define NUM_BLOCKS 1024
#define BLOCK_SIZE 256

// templated atomic_min
template<typename T>
inline T atomic_min(T *address, T val)
{
  sycl::atomic_ref<T, 
                   sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ao (*address);
  return ao.fetch_min(val);
}

// templated atomic_min
template<typename T>
inline T atomic_max(T *address, T val)
{
  sycl::atomic_ref<T, 
                   sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ao (*address);
  return ao.fetch_max(val);
}

// uint64 atomic add
template<typename T>
inline T atomic_add(T *address, T val)
{
  sycl::atomic_ref<T, 
                   sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ao (*address);
  return ao.fetch_add(val);
}

template <typename T>
void atomicMinDerived (sycl::nd_item<1> &item, T *__restrict res)
{
  int i = item.get_global_id(0) + 1;
  atomic_min(res, (T)i);
}

template <typename T>
void atomicMaxDerived (sycl::nd_item<1> &item, T *__restrict res)
{
  int i = item.get_global_id(0) + 1;
  atomic_max(res, (T)i);
}

template <typename T>
void atomicAddDerived (sycl::nd_item<1> &item, T *__restrict res)
{
  int i = item.get_global_id(0) + 1;
  atomic_add(res, (T)i);
}

template <typename T>
class k_atomic_min;

template <typename T>
class k_atomic_max;

template <typename T>
class k_atomic_add;

template <typename T>
void testMin (sycl::queue &q, T *h_ptr, T *d_ptr, const int repeat, const char* name) {
  sycl::range<1> gws (NUM_BLOCKS * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  q.memcpy(d_ptr, h_ptr, sizeof(T));

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k_atomic_min<T>>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        atomicMinDerived<T>(item, d_ptr);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic min for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(h_ptr, d_ptr, sizeof(T)).wait();
}

template <typename T>
void testMax (sycl::queue &q, T *h_ptr, T *d_ptr, const int repeat, const char* name) {
  sycl::range<1> gws (NUM_BLOCKS * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  q.memcpy(d_ptr, h_ptr, sizeof(T));

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k_atomic_max<T>>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        atomicMaxDerived<T>(item, d_ptr);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic max for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(h_ptr, d_ptr, sizeof(T)).wait();
}

template <typename T>
void testAdd (sycl::queue &q, T *h_ptr, T *d_ptr, const int repeat, const char* name) {
  sycl::range<1> gws (NUM_BLOCKS * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  q.memcpy(d_ptr, h_ptr, sizeof(T));

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k_atomic_add<T>>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        atomicAddDerived<T>(item, d_ptr);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic add for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(h_ptr, d_ptr, sizeof(T)).wait();
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  unsigned long long *d_res_u64 = sycl::malloc_device<unsigned long long>(3, q);
  long long *d_res_s64 = sycl::malloc_device<long long>(3, q);
  double *d_res_f64 = sycl::malloc_device<double>(3, q);

  testMin<unsigned long long>(q, res_u64, d_res_u64, repeat, "U64");
  testMin<long long>(q, res_s64, d_res_s64, repeat, "S64");
  testMin<double>(q, res_f64, d_res_f64, repeat, "F64");

  testMax<unsigned long long>(q, res_u64+1, d_res_u64+1, repeat, "U64");
  testMax<long long>(q, res_s64+1, d_res_s64+1, repeat, "S64");
  testMax<double>(q, res_f64+1, d_res_f64+1, repeat, "F64");

  // the add kernels are slow
  testAdd<unsigned long long>(q, res_u64+2, d_res_u64+2, 1, "U64");
  testAdd<long long>(q, res_s64+2, d_res_s64+2, 1, "S64");
  testAdd<double>(q, res_f64+2, d_res_f64+2, 1, "F64");

  unsigned long long sum = 0; 
  unsigned long long bound = NUM_BLOCKS*BLOCK_SIZE;
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

  sycl::free(d_res_u64, q);
  sycl::free(d_res_s64, q);
  sycl::free(d_res_f64, q);

  return 0;
}
