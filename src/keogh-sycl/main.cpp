#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

int main(int argc, char* argv[]) {

  if (argc != 4) {
    printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return -1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  // host side memory
  float *subject = (float*) malloc (sizeof(float)*N);
  float *lower = (float*) malloc (sizeof(float)*N);
  float *upper = (float*) malloc (sizeof(float)*N);
  float *lb = (float*) malloc (sizeof(float)*(N-M+1));
  float *lb_h = (float*) malloc (sizeof(float)*(N-M+1));
  float *avgs = (float*) malloc (sizeof(float)*(N-M+1));
  float *stds = (float*) malloc (sizeof(float)*(N-M+1));

  srand(123);
  for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) upper[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) lower[i] = (float)rand() / (float)RAND_MAX;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_subject = sycl::malloc_device<float>(N, q);
  q.memcpy(d_subject, subject, sizeof(float)*N);

  float *d_avgs = sycl::malloc_device<float>(N-M+1, q);
  q.memcpy(d_avgs, avgs, sizeof(float)*(N-M+1));

  float *d_stds = sycl::malloc_device<float>(N-M+1, q);
  q.memcpy(d_stds, stds, sizeof(float)*(N-M+1));

  float *d_lb = sycl::malloc_device<float>(N-M+1, q);

  float *d_lower = sycl::malloc_device<float>(N, q);
  q.memcpy(d_lower, lower, sizeof(float)*M);

  float *d_upper = sycl::malloc_device<float>(N, q);
  q.memcpy(d_upper, upper, sizeof(float)*M);

  const int blocks = 256;
  const int grids = (N-M+1 + blocks - 1) / blocks;
  sycl::range<1> gws (grids * blocks);
  sycl::range<1> lws (blocks);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> cache (sycl::range<1>(M+blocks), cgh);
      cgh.parallel_for<class lp_koegh>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int lid = item.get_local_id(0);
        int blockDim = item.get_local_range(0);
        int blockIdx = item.get_group(0);
        int blockSize = blockDim * blockIdx;
        int idx = blockSize + lid;

        for (int k = lid; k < blockDim + M; k += blockDim)
          if (blockSize + k < N) {
            cache[k] = d_subject[blockSize + k];
          }

        item.barrier(sycl::access::fence_space::local_space);

        if (idx < N-M+1) {

          // obtain statistics
          float residues = 0;
          float avg = d_avgs[idx];
          float std = d_stds[idx];

          for (int i = 0; i < M; ++i) {
            // differences to envelopes
            float value = (cache[lid+i] - avg) / std;
            float lower = value - d_lower[i];
            float upper = value - d_upper[i];

            // Euclidean distance
            residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
          }

          d_lb[idx] = residues;
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(lb, d_lb, sizeof(float)*(N-M+1)).wait();

  // verify
  reference(subject, avgs, stds, lb_h, lower, upper, M, N);
  bool ok = true;
  for (int i = 0; i < N-M+1; i++) {
    if (fabs(lb[i] - lb_h[i]) > 1e-2) {
      printf("%d %f %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  sycl::free(d_lb, q);
  sycl::free(d_avgs, q);
  sycl::free(d_stds, q);
  sycl::free(d_subject, q);
  sycl::free(d_lower, q);
  sycl::free(d_upper, q);
  free(lb);
  free(lb_h);
  free(avgs);
  free(stds);
  free(subject);
  free(lower);
  free(upper);
  return 0;
}


