/**
 * shmem_kernels.cu: This file is part of the gpumembench suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <chrono> // timing
#include <stdio.h>
#include <sycl/sycl.hpp>
#include "../shmembench-cuda/reference.h"

using namespace std::chrono;
using float4 = sycl::float4;

#define TOTAL_ITERATIONS (1024)
#define BLOCK_SIZE 256

// shared memory swap operation
void shmem_swap(float4 *v1, float4 *v2){
  float4 tmp;
  tmp = *v2;
  *v2 = *v1;
  *v1 = tmp;
}

float4 init_val(int i){
  return float4(i, i+11, i+19, i+23);
}

float4 reduce_vector(float4 v1, float4 v2, float4 v3, float4 v4, float4 v5, float4 v6){
  return (v1 + v2 + v3 + v4 + v5 + v6);
}

void set_vector(float4 *target, int offset, float4 v){
  target[offset] = v;
}


void benchmark_shmem(float4 *g_data, float4* shm_buffer, sycl::nd_item<1> &item){

  int tid = item.get_local_id(0);
  int blk = item.get_local_range(0);
  int gid = item.get_group(0);
  int globaltid = gid * blk + tid;

  set_vector(shm_buffer, tid+0*blk, init_val(tid));
  set_vector(shm_buffer, tid+1*blk, init_val(tid+1));
  set_vector(shm_buffer, tid+2*blk, init_val(tid+3));
  set_vector(shm_buffer, tid+3*blk, init_val(tid+7));
  set_vector(shm_buffer, tid+4*blk, init_val(tid+13));
  set_vector(shm_buffer, tid+5*blk, init_val(tid+17));

  item.barrier(sycl::access::fence_space::local_space);

  #pragma unroll 32
  for(int j=0; j<TOTAL_ITERATIONS; j++){
    shmem_swap(shm_buffer+tid+0*blk, shm_buffer+tid+1*blk);
    shmem_swap(shm_buffer+tid+2*blk, shm_buffer+tid+3*blk);
    shmem_swap(shm_buffer+tid+4*blk, shm_buffer+tid+5*blk);

    item.barrier(sycl::access::fence_space::local_space);

    shmem_swap(shm_buffer+tid+1*blk, shm_buffer+tid+2*blk);
    shmem_swap(shm_buffer+tid+3*blk, shm_buffer+tid+4*blk);

    item.barrier(sycl::access::fence_space::local_space);
  }

  g_data[globaltid] = reduce_vector(shm_buffer[tid+0*blk],
                                    shm_buffer[tid+1*blk],
                                    shm_buffer[tid+2*blk],
                                    shm_buffer[tid+3*blk],
                                    shm_buffer[tid+4*blk],
                                    shm_buffer[tid+5*blk]);
}

void shmembenchGPU(float *c, const size_t size, const int n) {
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // size floats, one float4 per work item => size/4 stored values
  if (size % ((size_t)BLOCK_SIZE * 4) != 0) {
    printf("size must be a multiple of %zu\n", (size_t)BLOCK_SIZE * 4);
    printf("FAIL\n");
    return;
  }
  const size_t num_float4 = size / 4;

  float *cd = sycl::malloc_device<float>(num_float4 * 4, q);

  sycl::range<1> lws (BLOCK_SIZE);
  sycl::range<1> gws (num_float4);

  auto start = high_resolution_clock::now();
  for (int i = 0; i < n; i++) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float4, 1> shm_buffer(sycl::range<1>(BLOCK_SIZE*6), cgh);
      cgh.parallel_for<class kernel>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        benchmark_shmem((float4*)cd,
                         shm_buffer.get_multi_ptr<sycl::access::decorated::no>().get(), item);
      });
    });
  }
  q.wait();
  auto end = high_resolution_clock::now();
  auto time_shmem_128b = duration_cast<nanoseconds>(end - start).count() / (double)n;
  printf("Average kernel execution time : %f (ms)\n", time_shmem_128b * 1e-6);

  // Copy results back to host memory
  q.memcpy(c, cd, num_float4 * sizeof(float4)).wait();
  sycl::free(cd, q);

  int errors = shmembench_verify(c, num_float4, BLOCK_SIZE, TOTAL_ITERATIONS);
  printf("%s\n", errors ? "FAIL" : "PASS");
  if (errors) return;

  printf("Memory throughput\n");
  // 6LL keeps the product in 64-bit; 20492*size overflows 32-bit int
  const long long nops = (6LL + 4 * 5 * TOTAL_ITERATIONS + 6) * size;
  const long long operations_bytes  = nops * sizeof(float);
  const long long operations_128bit = nops / 4;

  printf("\tusing 128bit operations : %8.2f GB/sec (%6.2f billion accesses/sec)\n",
    (double)operations_bytes / time_shmem_128b,
    (double)operations_128bit / time_shmem_128b);
}
