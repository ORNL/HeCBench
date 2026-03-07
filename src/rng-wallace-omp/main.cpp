#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <omp.h>
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
  float *globalPool = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);
  float *Pool_ref = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);

  for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++)
  {
    float x = RandN();
    Pool_ref[i] = globalPool[i] = x;
  }

  float* rngChi2Corrections = (float *) malloc(4 * WALLACE_CHI2_COUNT);
  for (unsigned int i = 0; i < WALLACE_CHI2_COUNT; i++)
  {
    rngChi2Corrections[i] = MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE);
  }
  float* randomNumbers = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);
  float* randomNumbers_ref = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);

  float *chi2Corrections = rngChi2Corrections;

  const unsigned seed = 1;

  #pragma omp target data map(to: globalPool[0: WALLACE_TOTAL_POOL_SIZE], \
                                  chi2Corrections[0:WALLACE_CHI2_COUNT]) \
                          map(alloc: randomNumbers[0: WALLACE_OUTPUT_SIZE])
  {
    // warmup and quick validation
    for (int i = 0; i < 30; i++) {
      rng_wallace(seed, globalPool, randomNumbers, chi2Corrections);
    }

    for (int i = 0; i < 30; i++) {
      reference(seed, Pool_ref, randomNumbers_ref, rngChi2Corrections, WALLACE_NUM_BLOCKS);
    }

    #pragma omp target update from(randomNumbers[0:WALLACE_OUTPUT_SIZE])
    #pragma omp target update from(globalPool[0:WALLACE_TOTAL_POOL_SIZE])

    bool ok = true;
    for (unsigned int n = 0; n < WALLACE_OUTPUT_SIZE; n++) {
      if (fabsf(randomNumbers_ref[n] - randomNumbers[n]) > 1e-3f) {
        printf("randNumbers mismatch at index %d: %f %f\n", n, randomNumbers_ref[n], randomNumbers[n]);
        ok = false;
        break;
      }
    }
    for (unsigned int n = 0; n < WALLACE_TOTAL_POOL_SIZE; n++) {
      if (fabsf(Pool_ref[n] - globalPool[n]) > 1e-3f) {
        printf("Pool mismatch at index %d: %f %f\n", n, Pool_ref[n], globalPool[n]);
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      rng_wallace(seed, globalPool, randomNumbers, chi2Corrections);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", time * 1e-3f / repeat);
  }

  free(rngChi2Corrections);
  free(randomNumbers);
  free(randomNumbers_ref);
  free(globalPool);
  free(Pool_ref);
  if (!ok) return 1;
  return 0;
}
