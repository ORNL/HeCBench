/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
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
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 256
#define INNER_ITERS 100
#define DEFAULT_LAUNCH_ITERATIONS 32

static double elapsed_us(std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration<double, std::micro>(end - start).count();
}

__global__ void reduce(const float *inputVec,
                       double *outputVec,
                       size_t inputSize, size_t outputSize)
{
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (size_t i = globaltid; i < inputSize; i += gridDim.x*blockDim.x)
  {
    temp_sum += (double) inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  double beta  = temp_sum;
  double temp;

  for (int i = tile32.size() / 2; i > 0; i >>= 1) {
    if (tile32.thread_rank() < i) {
      temp       = tmp[cta.thread_rank() + i];
      beta       += temp;
      tmp[cta.thread_rank()] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);

  if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {
    beta  = 0.0;
    for (int i = 0; i < cta.size(); i += tile32.size()) {
      beta  += tmp[i];
    }
    outputVec[blockIdx.x] =  beta;
  }
}

__global__ void reduceFinal(const double *inputVec,
                            double *result,
                            size_t inputSize)
{
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (size_t i = globaltid; i < inputSize; i += gridDim.x*blockDim.x)
  {
    temp_sum += inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  // do reduction in shared mem
  if ((blockDim.x >= 512) && (cta.thread_rank() < 256))
  {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 256];
  }

  cg::sync(cta);

  if ((blockDim.x >= 256) &&(cta.thread_rank() < 128))
  {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 128];
  }

  cg::sync(cta);

  if ((blockDim.x >= 128) && (cta.thread_rank() <  64))
  {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() +  64];
  }

  cg::sync(cta);

  if (cta.thread_rank() < 32)
  {
    // Fetch final intermediate sum from 2nd warp
    if (blockDim.x >=  64) temp_sum += tmp[cta.thread_rank() + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size()/2; offset > 0; offset /= 2)
    {
      temp_sum += tile32.shfl_down(temp_sum, offset);
    }
  }
  // write result for this block to global mem
  if (cta.thread_rank() == 0) result[0] = temp_sum;
}

void init_input(float*a, size_t size)
{
  srand(123);
  for (size_t i=0; i < size; i++)
    a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

static void usingStream(float *inputVec_h, float *inputVec_d, double *outputVec_d,
                        double *result_d, double *result_h, size_t inputSize,
                        size_t numOfBlocks, cudaStream_t stream)
{
  cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize,
                  cudaMemcpyHostToDevice, stream);

  for (int i = 0; i < INNER_ITERS; i++) {
    cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream);
    reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream>>>(
        inputVec_d, outputVec_d, inputSize, numOfBlocks);
    reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream>>>(outputVec_d, result_d,
                                                     numOfBlocks);
  }

  cudaMemcpyAsync(result_h, result_d, sizeof(double), cudaMemcpyDeviceToHost,
                  stream);
}

static void benchmark(float *inputVec_h, float *inputVec_d, double *outputVec_d,
                      double *result_d, size_t inputSize, size_t numOfBlocks,
                      int repeat)
{
  double result_r = 0.0;
  for (size_t i = 0; i < inputSize; i++)
    result_r += inputVec_h[i];

  double result_h = 0.0;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // 1. Eager warmup (do not print)
  usingStream(inputVec_h, inputVec_d, outputVec_d, result_d, &result_h, inputSize,
           numOfBlocks, stream);
  cudaStreamSynchronize(stream);

  // 2. Capture + instantiate (reported separately from replay)
  cudaGraph_t graph;
  auto t_cap0 = std::chrono::steady_clock::now();
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  usingStream(inputVec_h, inputVec_d, outputVec_d, result_d, &result_h, inputSize,
           numOfBlocks, stream);
  cudaStreamEndCapture(stream, &graph);
  cudaGraphExec_t graphExec;
  cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
  auto t_cap1 = std::chrono::steady_clock::now();
  printf("Capture+instantiate time: %f (us)\n", elapsed_us(t_cap0, t_cap1));

  // 3. One graph replay (do not print)
  cudaGraphLaunch(graphExec, stream);
  cudaStreamSynchronize(stream);

  // 4. Timed graph replays
  auto t_g0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
  }
  auto t_g1 = std::chrono::steady_clock::now();
  printf("%s\n", (std::fabs(result_h - result_r) < 1e-6) ? "PASS" : "FAIL");
  printf("Average execution time of using Graph: %f (us)\n\n",
         elapsed_us(t_g0, t_g1) / repeat);

  // 5. Timed eager (kernels already hot)
  auto t_s0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    usingStream(inputVec_h, inputVec_d, outputVec_d, result_d, &result_h, inputSize,
             numOfBlocks, stream);
    cudaStreamSynchronize(stream);
  }
  auto t_s1 = std::chrono::steady_clock::now();
  printf("%s\n", (std::fabs(result_h - result_r) < 1e-6) ? "PASS" : "FAIL");
  printf("Average execution time of using Stream: %f (us)\n\n",
         elapsed_us(t_s0, t_s1) / repeat);

  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

int main(int argc, char **argv)
{
  int repeat = DEFAULT_LAUNCH_ITERATIONS;
  if (argc == 2) {
    repeat = atoi(argv[1]);
  } else if (argc > 2) {
    printf("Usage: %s [repeat]\n", argv[0]);
    return 1;
  }
  if (repeat < 1) {
    printf("repeat must be >= 1\n");
    return 1;
  }

  size_t maxBlocks = 512;

  for (size_t size = 512; size <= 1 << 27; size = size * 512) {

    printf("\n-----------------------------\n");
    printf("%zu elements\n", size);
    printf("Threads per block  = %d\n", THREADS_PER_BLOCK);
    printf("Launch iterations = %d\n", repeat);

    float *inputVec_d = NULL, *inputVec_h = NULL;
    double *outputVec_d = NULL, *result_d;

    inputVec_h = (float *)malloc(sizeof(float) * size);
    cudaMalloc(&inputVec_d, sizeof(float) * size);
    cudaMalloc(&outputVec_d, sizeof(double) * maxBlocks);
    cudaMalloc(&result_d, sizeof(double));

    init_input(inputVec_h, size);

    benchmark(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks,
              repeat);

    cudaFree(inputVec_d);
    cudaFree(outputVec_d);
    cudaFree(result_d);
    free(inputVec_h);
  }
  return EXIT_SUCCESS;
}
