#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

#include "rand_helpers.h"
#include "constants.h"
#include "wallace_kernel.h"
#include "reference.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // host buffers
  float *hostPool = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);
  float *Pool_ref = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);

  for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++) {
    float x = RandN();
    Pool_ref[i] = hostPool[i] = x;
  }

  float* rngChi2Corrections = (float *) malloc(4 * WALLACE_CHI2_COUNT);
  for (unsigned int i = 0; i < WALLACE_CHI2_COUNT; i++)
  {
    rngChi2Corrections[i] = MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE);
  }
  float* randomNumbers = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);
  float* randomNumbers_ref = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);

  // device buffers
  float *device_randomNumbers;
  float *devicerngChi2Corrections;
  float *devPool;
  cudaMalloc((void **) &devPool, 4 * WALLACE_TOTAL_POOL_SIZE);
  cudaMalloc((void **) &devicerngChi2Corrections, 4 * WALLACE_CHI2_COUNT);
  cudaMalloc((void **) &device_randomNumbers, 4 * WALLACE_OUTPUT_SIZE);

  cudaMemcpy(devPool, hostPool, 4 * WALLACE_TOTAL_POOL_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(devicerngChi2Corrections, rngChi2Corrections,
             4 * WALLACE_CHI2_COUNT, cudaMemcpyHostToDevice);

  dim3 rng_wallace_grid(WALLACE_NUM_BLOCKS, 1, 1);
  dim3 rng_wallace_threads(WALLACE_NUM_THREADS, 1, 1);
  const unsigned seed = 1;

  // warmup and quick validation
  for (int i = 0; i < 30; i++) {
    rng_wallace <<< rng_wallace_grid, rng_wallace_threads >>> (
        seed, devPool, device_randomNumbers, devicerngChi2Corrections);
  }

  for (int i = 0; i < 30; i++) {
    reference(seed, Pool_ref, randomNumbers_ref, rngChi2Corrections, WALLACE_NUM_BLOCKS);
  }

  cudaMemcpy(randomNumbers, device_randomNumbers,
             4 * WALLACE_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostPool, devPool,
             4 * WALLACE_TOTAL_POOL_SIZE, cudaMemcpyDeviceToHost);

  bool ok = true;
  for (unsigned int n = 0; n < WALLACE_OUTPUT_SIZE; n++) {
    if (fabsf(randomNumbers_ref[n] - randomNumbers[n]) > 1e-3f) {
      printf("randNumbers mismatch at index %d: %f %f\n", n, randomNumbers_ref[n], randomNumbers[n]);
      ok = false;
      break;
    }
  }
  for (unsigned int n = 0; n < WALLACE_TOTAL_POOL_SIZE; n++) {
    if (fabsf(Pool_ref[n] - hostPool[n]) > 1e-3f) {
      printf("Pool mismatch at index %d: %f %f\n", n, Pool_ref[n], hostPool[n]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    rng_wallace <<< rng_wallace_grid, rng_wallace_threads >>> (
        seed, devPool, device_randomNumbers, devicerngChi2Corrections);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", time * 1e-3f / repeat);

  free(rngChi2Corrections);
  free(randomNumbers);
  free(randomNumbers_ref);
  free(hostPool);
  free(Pool_ref);
  cudaFree(devicerngChi2Corrections);
  cudaFree(device_randomNumbers);
  cudaFree(devPool);
  if (!ok) return 1;
  return 0;
}
