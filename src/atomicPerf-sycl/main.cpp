#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define BLOCK_SIZE 256

#include "reference.h"

template <typename T>
void BlockRangeAtomicOnGlobalMem(T* data, int n, sycl::nd_item<1> &item)
{
  unsigned int tid = item.get_global_id(0);
  for (unsigned int i = tid; i < n;
       i += item.get_local_range(0) * item.get_group_range(0)) {
    auto ao = sycl::atomic_ref<T, 
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space> (data[item.get_local_id(0)]);
    ao.fetch_add((T)1); // arbitrary number to add
  }
}

template <typename T>
void WarpRangeAtomicOnGlobalMem(T* data, int n, sycl::nd_item<1> &item)
{
  unsigned int tid = item.get_global_id(0);
  for (unsigned int i = tid; i < n;
       i += item.get_local_range(0) * item.get_group_range(0)) {
    auto ao = sycl::atomic_ref<T, 
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space> (data[i & 0x1F]);
    ao.fetch_add((T)1); // arbitrary number to add
  }
}

template <typename T>
void SingleRangeAtomicOnGlobalMem(T* data, int offset, int n, sycl::nd_item<1> &item)
{
  unsigned int tid = item.get_global_id(0);
  for (unsigned int i = tid; i < n;
       i += item.get_local_range(0) * item.get_group_range(0)) {
    auto ao = sycl::atomic_ref<T, 
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space> (data[offset]);
    ao.fetch_add((T)1); // arbitrary number to add
  }
}

template <typename T>
void BlockRangeAtomicOnSharedMem(T* data, int n, sycl::nd_item<1> item,
                                 T *smem_data)
{
  unsigned int tid = item.get_global_id(0);
  for (unsigned int i = tid; i < n;
       i += item.get_local_range(0) * item.get_group_range(0)) {
    auto ao = sycl::atomic_ref<T, 
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space> (smem_data[item.get_local_id(0)]);
    ao.fetch_add((T)1); // arbitrary number to add
  }
  if (item.get_group(0) == item.get_group_range(0))
    data[item.get_local_id(0)] = smem_data[item.get_local_id(0)];
}

template <typename T>
void WarpRangeAtomicOnSharedMem(T* data, int n, sycl::nd_item<1> item,
                                T *smem_data)
{
  unsigned int tid = item.get_global_id(0);
  for (unsigned int i = tid; i < n;
       i += item.get_local_range(0) * item.get_group_range(0)) {
    auto ao = sycl::atomic_ref<T, 
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space> (smem_data[i & 0x1F]);
    ao.fetch_add((T)1); // arbitrary number to add
  }
  if (item.get_group(0) == item.get_group_range(0) &&
      item.get_local_id(0) < 0x1F)
    data[item.get_local_id(0)] = smem_data[item.get_local_id(0)];
}

template <typename T>
void SingleRangeAtomicOnSharedMem(T* data, int offset, int n, sycl::nd_item<1> item,
                                  T *smem_data)
{
  unsigned int tid = item.get_global_id(0);
  for (unsigned int i = tid; i < n;
       i += item.get_local_range(0) * item.get_group_range(0)) {
    auto ao = sycl::atomic_ref<T, 
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space> (smem_data[offset]);
    ao.fetch_add((T)1); // arbitrary number to add
  }
  if (item.get_group(0) == item.get_group_range(0) &&
      item.get_local_id(0) == 0)
    data[item.get_local_id(0)] = smem_data[item.get_local_id(0)];
}

template <typename T>
void atomicPerf (int n, int t, int repeat)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  
  size_t data_size = sizeof(T) * t;

  T* data = (T*) malloc (data_size);
  T* h_data = (T*) malloc (data_size);
  T* r_data = (T*) malloc (data_size);
  int fail;

  for(int i=0; i<t; i++) {
    data[i] = i%1024+1;
  }

  T* d_data = (T *)sycl::malloc_device(data_size, q);

  sycl::range<1> lws (BLOCK_SIZE);
  sycl::range<1> gws (n);

  q.memcpy(d_data, data, data_size).wait();
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        BlockRangeAtomicOnGlobalMem<T>(d_data, n, item);
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    BlockRangeAtomicOnGlobalMem_ref<T>(r_data, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");
  
  q.memcpy(d_data, data, data_size).wait();
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        WarpRangeAtomicOnGlobalMem<T>(d_data, n, item);
      });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WarpRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    WarpRangeAtomicOnGlobalMem_ref<T>(r_data, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  q.memcpy(d_data, data, data_size).wait();
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        SingleRangeAtomicOnGlobalMem<T>(d_data, i % BLOCK_SIZE, n, item);
      });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    SingleRangeAtomicOnGlobalMem_ref<T>(r_data, i % BLOCK_SIZE, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  q.memcpy(d_data, data, data_size).wait();
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<T, 1> smem (sycl::range<1>(BLOCK_SIZE), cgh);
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        BlockRangeAtomicOnSharedMem<T>(d_data, n, item, smem.template get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  q.memcpy(d_data, data, data_size).wait();
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<T, 1> smem (sycl::range<1>(32), cgh);
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        WarpRangeAtomicOnSharedMem<T>(d_data, n, item, smem.template get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WarpRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  q.memcpy(d_data, data, data_size).wait();
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<T, 1> smem (sycl::range<1>(BLOCK_SIZE), cgh);
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        SingleRangeAtomicOnSharedMem<T>(d_data, i % BLOCK_SIZE,
                                        n, item, smem.template get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  free(data);
  free(h_data);
  free(r_data);
  sycl::free(d_data, q);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int n = 3*4*7*8*9*256; // number of threads
  const int len = 1024; // data array length
  
  printf("\nFP64 atomic add\n");
  atomicPerf<double>(n, len, repeat); 

  printf("\nINT32 atomic add\n");
  atomicPerf<int>(n, len, repeat); 

  printf("\nFP32 atomic add\n");
  atomicPerf<float>(n, len, repeat); 

  if (fail) return 1;
  if (fail) return 1;
  if (fail) return 1;
  if (fail) return 1;
  if (fail) return 1;
  if (fail) return 1;
  return 0;
}
