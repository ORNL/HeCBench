#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
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
  for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++)
  {
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_Pool= sycl::malloc_device<float>(WALLACE_TOTAL_POOL_SIZE, q);
  float *d_rngChi2Corrections= sycl::malloc_device<float>(WALLACE_CHI2_COUNT, q);
  float *d_randomNumbers= sycl::malloc_device<float>(WALLACE_OUTPUT_SIZE, q);

  q.memcpy(d_Pool, hostPool, sizeof(float) * WALLACE_TOTAL_POOL_SIZE);
  q.memcpy(d_rngChi2Corrections, rngChi2Corrections, sizeof(float) * WALLACE_CHI2_COUNT);

  sycl::range<1> gws (WALLACE_NUM_BLOCKS * WALLACE_NUM_THREADS);
  sycl::range<1> lws (WALLACE_NUM_THREADS);
  const unsigned seed = 1;

  auto kFn = [&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> pool (
      sycl::range<1>(WALLACE_POOL_SIZE + WALLACE_CHI2_SHARED_SIZE), cgh);
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      rng_wallace(seed, d_Pool, d_randomNumbers, d_rngChi2Corrections,
                  item, pool.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  };

  // warmup and quick validation
  for (int i = 0; i < 30; i++) {
    q.submit(kFn);
  }

  for (int i = 0; i < 30; i++) {
    reference(seed, Pool_ref, randomNumbers_ref, rngChi2Corrections, WALLACE_NUM_BLOCKS);
  }
  q.memcpy(randomNumbers, d_randomNumbers, 4 * WALLACE_OUTPUT_SIZE);
  q.memcpy(hostPool, d_Pool, 4 * WALLACE_TOTAL_POOL_SIZE);
  q.wait();

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

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit(kFn);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", time * 1e-3f / repeat);

  free(rngChi2Corrections);
  free(randomNumbers);
  free(randomNumbers_ref);
  free(hostPool);
  free(Pool_ref);
  sycl::free(d_rngChi2Corrections, q);
  sycl::free(d_randomNumbers, q);
  sycl::free(d_Pool, q);
  return 0;
}
