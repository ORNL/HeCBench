#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <sycl/sycl.hpp>
#include "kernels.h"

#define CSV 0
#if(CSV)
#define PS(X, S) std::cout << X << ", " << S << ", "; fflush(stdout);
#define PV(X) std::cout << X << ", "; fflush(stdout);
#else
#define PS(X, S) std::cout << X << " " << S <<" :\n"; fflush(stdout);
#define PV(X) std::cout << "\t" << #X << " \t: " << X << "\n"; fflush(stdout);
#endif

/*
 * Produce 64-bits of pseudo-randomness
 * Note: not very "good" or "random"
 */
template<typename vec_t>
vec_t rand64() {
  vec_t rtn;
  do {
    uint32_t * rtn32 = (uint32_t *)&rtn;
    rtn32[0] = rand();
    if(sizeof(vec_t) > 4) rtn32[1] = rand();
  } while(!(rtn < getPositiveInfinity<vec_t>() &&
        rtn > getNegativeInfinity<vec_t>()));
  return rtn;
}

template<typename vec_t, uint32_t blocks, uint32_t threads, bool timing>
class divide;

template<typename vec_t, uint32_t blocks, uint32_t threads, bool timing>
class merge;

/*
 * Perform <runs> merges of two sorted pseudorandom <vec_t> arrays of length <size>
 * Checks the output of each merge for correctness
 */
#define PADDING 1024
template<typename vec_t, uint32_t blocks, uint32_t threads, bool timing>
void mergeType(sycl::queue &q, const uint64_t size, const uint32_t runs) {
  // Prepare host and device vectors
  std::vector<vec_t> hA (size + PADDING);
  std::vector<vec_t> hB (size + PADDING);
  std::vector<vec_t> hC (2*size + PADDING);

  vec_t *dA = sycl::malloc_device<vec_t>(size + PADDING, q);
  vec_t *dB = sycl::malloc_device<vec_t>(size + PADDING, q);
  vec_t *dC = sycl::malloc_device<vec_t>(2*size + PADDING, q);

  // diagonal_path_intersections;
  uint32_t *dpi = sycl::malloc_device<uint32_t>(2 * (blocks + 1), q);

  uint32_t errors = 0;

  double total_time = 0.0;

  for(uint32_t r = 0; r < runs; r++) {

    // Generate two sorted psuedorandom arrays
    for (uint64_t n = 0; n < size; n++) {
       hA[n] = rand64<vec_t>();
       hB[n] = rand64<vec_t>();
    }

    for (uint64_t n = size; n < size + PADDING; n++) {
      hA[n] = getPositiveInfinity<vec_t>();
      hB[n] = getPositiveInfinity<vec_t>();
    }

    std::sort(hA.begin(), hA.end());
    std::sort(hB.begin(), hB.end());

    q.memcpy(dA, hA.data(), (size + PADDING) * sizeof(vec_t));
    q.memcpy(dB, hB.data(), (size + PADDING) * sizeof(vec_t));

    // Perform the global diagonal intersection serach to divide work among SMs
    sycl::range<1> gws (blocks * 32);
    sycl::range<1> lws (32);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<int32_t, 0> xt(cgh);
      sycl::local_accessor<int32_t, 0> yt(cgh);
      sycl::local_accessor<int32_t, 0> xb(cgh);
      sycl::local_accessor<int32_t, 0> yb(cgh);
      sycl::local_accessor<int32_t, 0> found(cgh);
      sycl::local_accessor<int32_t, 1> oneorzero(sycl::range<1>(32), cgh);
      cgh.parallel_for<class divide<vec_t, blocks, threads, timing>>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        workloadDiagonals<vec_t>(
          item, xt, yt, xb, yb,
          found, oneorzero.get_multi_ptr<sycl::access::decorated::no>().get(),
          dA, size, dB, size, dpi);
      });
    });

    // Merge between global diagonals independently on each block
    sycl::range<1> gws2 (blocks * threads);
    sycl::range<1> lws2 (threads);
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<vec_t, 1> A(sycl::range<1>((K+2)<<1), cgh);
      sycl::local_accessor<uint32_t, 0> xt(cgh);
      sycl::local_accessor<uint32_t, 0> yt(cgh);
      sycl::local_accessor<uint32_t, 0> xs(cgh);
      sycl::local_accessor<uint32_t, 0> ys(cgh);
      cgh.parallel_for<class merge<vec_t, blocks, threads, timing>>(
        sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
        mergeSinglePath<vec_t,false,false> (
          item, A.template get_multi_ptr<sycl::access::decorated::no>().get(), xt, yt, xs, ys,
          dA, size, dB, size,
          dpi, dC, size * 2);
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    // Test for errors
    q.memcpy(hC.data(), dC, size * sizeof(vec_t)).wait();

    for(uint32_t i = 1; i < size; i++) {
      errors += hC[i] < hC[i-1];
    }
  }

  sycl::free(dA, q);
  sycl::free(dB, q);
  sycl::free(dC, q);
  sycl::free(dpi, q);

  PV(errors); // Print error info
  printf("%s. ", errors ? "FAIL" : "PASS");
  if (errors) exit(1);

  if (timing)
    printf("Average kernel execution time: %f (us).\n", (total_time * 1e-3f) / runs);
  else
    printf("Warmup run\n");
}

/*
 * Performs <runs> merge tests for each type at a given size
 */
template<uint32_t blocks, uint32_t threads>
void mergeAllTypes(sycl::queue &q, const uint64_t size, const uint32_t runs) {
  PS("uint32_t", size)  mergeType<uint32_t, blocks, threads, false>(q, size, runs); printf("\n");
  PS("uint32_t", size)  mergeType<uint32_t, blocks, threads, true>(q, size, runs); printf("\n");

  PS("float",    size)  mergeType<float,    blocks, threads, false>(q, size, runs); printf("\n");
  PS("float",    size)  mergeType<float,    blocks, threads, true>(q, size, runs); printf("\n");

  PS("uint64_t", size)  mergeType<uint64_t, blocks, threads, false>(q, size, runs); printf("\n");
  PS("uint64_t", size)  mergeType<uint64_t, blocks, threads, true>(q, size, runs); printf("\n");

  PS("double", size)    mergeType<double,   blocks, threads, false>(q, size, runs); printf("\n");
  PS("double", size)    mergeType<double,   blocks, threads, true>(q, size, runs); printf("\n");
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length of the arrays> <runs>\n", argv[0]);
    return 1;
  }
  // length is sufficiently large;
  // otherwise there are invalid global reads in the kernel mergeSinglePath
  const uint64_t length = atol(argv[1]);

  const uint32_t runs = atoi(argv[2]);

  const int blocks = 112;
  const int threads = 128;  // do not change

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  mergeAllTypes<blocks, threads>(q, length, runs);

  return 0;
}

