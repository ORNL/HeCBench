/**
 * shmem_kernels.cu: This file is part of the gpumembench suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <chrono> // timing
#include <stdio.h>
#include "hip/hip_runtime.h"

using namespace std::chrono;

#define TOTAL_ITERATIONS (1024)
#define BLOCK_SIZE 256

// shared memory swap operation
__device__ void shmem_swap(float4 *v1, float4 *v2){
  float4 tmp;
  tmp = *v2;
  *v2 = *v1;
  *v1 = tmp;
}

__device__ float4 init_val(int i){
  return make_float4(i, i+11, i+19, i+23);
}

__device__ float4 reduce_vector(float4 v1, float4 v2, float4 v3, float4 v4, float4 v5, float4 v6){
  return (v1 + v2 + v3 + v4 + v5 + v6);
}

__device__ void set_vector(float4 *target, int offset, float4 v){
  target[offset] = v;
}


__global__ void benchmark_shmem(float4 *g_data){

  __shared__ float4 shm_buffer[BLOCK_SIZE*6];

  int tid = threadIdx.x; 
  int globaltid = blockIdx.x*blockDim.x + tid;
  set_vector(shm_buffer, tid+0*blockDim.x, init_val(tid));
  set_vector(shm_buffer, tid+1*blockDim.x, init_val(tid+1));
  set_vector(shm_buffer, tid+2*blockDim.x, init_val(tid+3));
  set_vector(shm_buffer, tid+3*blockDim.x, init_val(tid+7));
  set_vector(shm_buffer, tid+4*blockDim.x, init_val(tid+13));
  set_vector(shm_buffer, tid+5*blockDim.x, init_val(tid+17));

  __syncthreads();  // __threadfence_block() is faster though

  #pragma unroll 32
  for(int j=0; j<TOTAL_ITERATIONS; j++){
    shmem_swap(shm_buffer+tid+0*blockDim.x, shm_buffer+tid+1*blockDim.x);
    shmem_swap(shm_buffer+tid+2*blockDim.x, shm_buffer+tid+3*blockDim.x);
    shmem_swap(shm_buffer+tid+4*blockDim.x, shm_buffer+tid+5*blockDim.x);

    __syncthreads();

    shmem_swap(shm_buffer+tid+1*blockDim.x, shm_buffer+tid+2*blockDim.x);
    shmem_swap(shm_buffer+tid+3*blockDim.x, shm_buffer+tid+4*blockDim.x);

    __syncthreads();
  }

  g_data[globaltid] = reduce_vector(shm_buffer[tid+0*blockDim.x], 
                                    shm_buffer[tid+1*blockDim.x],
                                    shm_buffer[tid+2*blockDim.x],
                                    shm_buffer[tid+3*blockDim.x],
                                    shm_buffer[tid+4*blockDim.x],
                                    shm_buffer[tid+5*blockDim.x]);
}

void shmembenchGPU(double *c, const long size, const int n) {
  const int TOTAL_BLOCKS = size/(BLOCK_SIZE);

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid_f4(TOTAL_BLOCKS/4, 1, 1);

  // Every thread stores exactly one float4, so this is the whole extent the
  // kernel touches. Sizing the buffer to it keeps the allocation, the copy back
  // and the checksum in agreement; otherwise the tail is never written and the
  // checksum depends on whatever the allocator happens to return.
  const size_t buffer_bytes = (size_t)dimGrid_f4.x * BLOCK_SIZE * sizeof(float4);

  double *cd;
  hipMalloc((void**)&cd, buffer_bytes);

  auto start = high_resolution_clock::now();

  for (int i = 0; i < n; i++)
    benchmark_shmem<<< dimGrid_f4, dimBlock >>>((float4*)cd);

  hipDeviceSynchronize();
  auto end = high_resolution_clock::now();
  auto time_shmem_128b = duration_cast<nanoseconds>(end - start).count() / (double)n;
  printf("Average kernel execution time : %f (ms)\n", time_shmem_128b * 1e-6);

  // Copy results back to host memory
  hipMemcpy(c, cd, buffer_bytes, hipMemcpyDeviceToHost);
  hipFree(cd);

  // simple checksum
  double sum = 0;
  for (size_t i = 0; i < buffer_bytes/sizeof(double); i++) sum += c[i];
  if (sum != 21256458760384741137729978368.00)
    printf("checksum failed\n");
  
  printf("Memory throughput\n");
  const long long operations_bytes  = (6LL+4*5*TOTAL_ITERATIONS+6)*size*sizeof(float);
  const long long operations_128bit = (6LL+4*5*TOTAL_ITERATIONS+6)*size/4;

  printf("\tusing 128bit operations : %8.2f GB/sec (%6.2f billion accesses/sec)\n", 
    (double)operations_bytes / time_shmem_128b,
    (double)operations_128bit / time_shmem_128b);
}
