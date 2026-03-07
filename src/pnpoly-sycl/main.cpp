#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <sycl/sycl.hpp>

#define VERTICES 600
#define BLOCK_SIZE_X 256

#include "kernel.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int nPoints = 2e7;
  const int vertices = VERTICES;

  std::default_random_engine rng (123);
  std::normal_distribution<float> distribution(0, 1);

  float2 *point = (float2*) malloc (sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    point[i].x() = distribution(rng);
    point[i].y() = distribution(rng);
  }

  float2 *vertex = (float2*) malloc (vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x() = cosf(t);
    vertex[i].y() = sinf(t);
  }

  // kernel results
  int *bitmap_ref = (int*) malloc (nPoints * sizeof(int));
  int *bitmap_opt = (int*) malloc (nPoints * sizeof(int));

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float2 *d_point = sycl::malloc_device<float2>(nPoints, q);
  float2 *d_vertex = sycl::malloc_device<float2>(vertices, q);
  q.memcpy(d_point, point, nPoints*sizeof(float2));
  q.memcpy(d_vertex, vertex, vertices*sizeof(float2));

  int *d_bitmap_ref = sycl::malloc_device<int>(nPoints, q);
  int *d_bitmap_opt = sycl::malloc_device<int>(nPoints, q);

  //kernel parameters
  sycl::range<1> lws (BLOCK_SIZE_X);
  sycl::range<1> gws ((nPoints+BLOCK_SIZE_X-1) / BLOCK_SIZE_X * BLOCK_SIZE_X);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class reference>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        pnpoly_base(item, d_bitmap_ref, d_point, d_vertex, nPoints);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(bitmap_ref, d_bitmap_ref, nPoints*sizeof(int)).wait();

  // performance tuning with tile sizes
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class opt1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        pnpoly_opt<1>(item, d_bitmap_opt, d_point, d_vertex, nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class opt2>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        pnpoly_opt<2>(item, d_bitmap_opt, d_point, d_vertex, nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class opt3>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        pnpoly_opt<4>(item, d_bitmap_opt, d_point, d_vertex, nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class opt4>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        pnpoly_opt<8>(item, d_bitmap_opt, d_point, d_vertex, nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class opt5>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        pnpoly_opt<16>(item, d_bitmap_opt, d_point, d_vertex, nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class opt6>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        pnpoly_opt<32>(item, d_bitmap_opt, d_point, d_vertex, nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class opt7>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        pnpoly_opt<64>(item, d_bitmap_opt, d_point, d_vertex, nPoints);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(bitmap_opt, d_bitmap_opt, nPoints*sizeof(int)).wait();

  // compare against reference kernel for verification
  int error = memcmp(bitmap_opt, bitmap_ref, nPoints*sizeof(int)); 

  // double check
  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  sycl::free(d_vertex, q);
  sycl::free(d_point, q);
  sycl::free(d_bitmap_ref, q);
  sycl::free(d_bitmap_opt, q);

  free(vertex);
  free(point);
  free(bitmap_ref);
  free(bitmap_opt);
  return 0;
}
