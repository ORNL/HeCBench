#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

__global__ void sum (
    const float*__restrict__ array,
    const int N,
    unsigned int *__restrict__ count,
    volatile float*__restrict__ result)
{
  __shared__ bool isLastBlockDone;
  __shared__ float partialSum;

  // Each block sums a subset of the input array.
  unsigned int bid = blockIdx.x;
  unsigned int num_blocks = gridDim.x;
  unsigned int block_size = blockDim.x;
  unsigned int lid = threadIdx.x;
  unsigned int gid = bid * block_size + lid;

  if (lid == 0) partialSum = 0;
  __syncthreads();

  if (gid < N)
    atomicAdd(&partialSum, array[gid]);

  __syncthreads();

  if (lid == 0) {

    // Thread 0 of each block stores the partial sum
    // to global memory. The compiler will use 
    // a store operation that bypasses the L1 cache
    // since the "result" variable is declared as
    // volatile. This ensures that the threads of
    // the last block will read the correct partial
    // sums computed by all other blocks.
    result[bid] = partialSum;

    // Thread 0 makes sure that the incrementation
    // of the "count" variable is only performed after
    // the partial sum has been written to global memory.
    __threadfence();

    // Thread 0 signals that it is done.
    unsigned int value = atomicAdd(count, 1);

    // Thread 0 determines if its block is the last
    // block to be done.
    isLastBlockDone = (value == (num_blocks - 1));
  }

  // Synchronize to make sure that each thread reads
  // the correct value of isLastBlockDone.
  __syncthreads();

  if (isLastBlockDone) {

    // The last block sums the partial sums
    // stored in result[0 .. num_blocks-1]
    if (lid == 0) partialSum = 0;
    __syncthreads();

    for (int i = lid; i < num_blocks; i += block_size)
      atomicAdd(&partialSum, result[i]);

    __syncthreads();

    if (lid == 0) {
      // Thread 0 of last block stores the total sum
      // to global memory and resets the count
      // varialble, so that the next kernel call
      // works properly.
      result[0] = partialSum;
      *count = 0;
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <repeat> <array length>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int N = atoi(argv[2]);

  const int blocks = 256;
  const int grids = (N + blocks - 1) / blocks;

  float* h_array = (float*) malloc (N * sizeof(float));
  float h_sum;

  float* d_result;
  cudaMalloc((void**)&d_result, grids * sizeof(float));

  float* d_array;
  cudaMalloc((void**)&d_array, N * sizeof(float));

  unsigned int* d_count;
  cudaMalloc((void**)&d_count, sizeof(unsigned int));
  cudaMemset(d_count, 0u, sizeof(unsigned int));

  bool ok = true;
  double time = 0.0;

  for (int i = 0; i < N; i++) h_array[i] = -1.f;
  cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  for (int n = 0; n < repeat; n++) {
    auto start = std::chrono::steady_clock::now();

    sum <<< grids, blocks >>> (d_array, N, d_count, d_result);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  cudaMemcpy(&h_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost);
  if (h_sum != -1.f * N) {
    ok = false;
  }

  if (ok) printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  free(h_array);
  cudaFree(d_result);
  cudaFree(d_array);
  cudaFree(d_count);

  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
  return 0;
}
