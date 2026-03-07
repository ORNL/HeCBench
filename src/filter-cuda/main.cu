/**
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include <cuda.h>
#include <cooperative_groups.h>

__global__ 
void filter (int *__restrict__ dst,
             int *__restrict__ nres,
             const int*__restrict__ src,
             int n)
{
  __shared__ int l_n;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // zero the counter
  if (threadIdx.x == 0)
    l_n = 0;
  __syncthreads();

  // get the value, evaluate the predicate, and
  // increment the counter if needed
  int d, pos;

  if(i < n) {
    d = src[i];
    if(d > 0)
      pos = atomicAdd(&l_n, 1);
  }
  __syncthreads();

  // leader increments the global counter
  if(threadIdx.x == 0)
    l_n = atomicAdd(nres, l_n);
  __syncthreads();

  // threads with true predicates write their elements
  if(i < n && d > 0) {
    pos += l_n; // increment local pos by global counter
    dst[pos] = d;
  }
  __syncthreads();
}


__device__ int atomicAggInc(int *ctr) {
  auto g = cooperative_groups::coalesced_threads();
  int warp_res = 0;
  if(g.thread_rank() == 0)
    warp_res = atomicAdd(ctr, g.size());
  return g.shfl(warp_res, 0) + g.thread_rank();
}

__global__
void filter2 (int *__restrict__ dst,
              int *__restrict__ nres,
              const int*__restrict__ src,
              int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n && src[i] > 0)
    dst[atomicAggInc(nres)] = src[i];
}


// compare device results with host results
bool check(int *d_nres, int *d_output, int h_nres, std::vector<int> &h_output) {
  int nres;
  cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<int> output (nres);

  cudaMemcpy(output.data(), d_output, sizeof(int) * nres, cudaMemcpyDeviceToHost);

  // clear device output
  cudaMemset(d_output, 0, sizeof(int) * nres);

  std::sort(output.begin(), output.end());

  bool equal = (h_nres == nres) && 
               std::equal(h_output.begin(),
                          h_output.begin() + h_nres, output.begin());
  return equal;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <number of elements> <block size> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_elems = atoi(argv[1]);
  const int block_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
    
  std::vector<int> input (num_elems);

  // Generate input data.
  for (int i = 0; i < num_elems; i++) {
    input[i] = i - num_elems / 2;
  }

  std::mt19937 g;
  g.seed(19937);
  std::shuffle(input.begin(), input.end(), g);

  // Generate host output with host filtering code.
  std::vector<int> h_output (num_elems);

  int h_flt_count = 0;
  for (int i = 0; i < num_elems; i++) {
    if (input[i] > 0) {
      h_output[h_flt_count++] = input[i];
    }
  }
  // Sort the result for comparison
  std::sort(h_output.begin(), h_output.begin() + h_flt_count);

  // Filtering on a device
  int *d_input, *d_output, *d_nres;

  cudaMalloc(&d_input, sizeof(int) * num_elems);
  cudaMalloc(&d_output, sizeof(int) * num_elems);
  cudaMalloc(&d_nres, sizeof(int));

  cudaMemcpy(d_input, input.data(),
             sizeof(int) * num_elems, cudaMemcpyHostToDevice);

  dim3 dimBlock (block_size);
  dim3 dimGrid ((num_elems + block_size - 1) / block_size);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_nres, 0, sizeof(int));
    filter<<<dimGrid, dimBlock>>>(d_output, d_nres, d_input, num_elems);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of filter (shared memory) %lf (ms)\n",
         (time * 1e-6) / repeat);

  bool match = check(d_nres, d_output, h_flt_count, h_output);
  printf("%s\n", match ? "PASS" : "FAIL");
  if (!match) exit(1);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_nres, 0, sizeof(int));
    filter2<<<dimGrid, dimBlock>>>(d_output, d_nres, d_input, num_elems);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of filter (global aggregate) %lf (ms)\n",
         (time * 1e-6) / repeat);

  match = check(d_nres, d_output, h_flt_count, h_output);
  printf("%s\n", match ? "PASS" : "FAIL");
  if (!match) exit(1);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_nres);

  return 0;
}
