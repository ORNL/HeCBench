#include <algorithm>
#include <chrono> // for high_resolution_clock
#include <cstdio>
#include <random>

#include "gpu_solver.h"
#include "reference.h"
#include <cstdlib>

void generate_data(int size, int min, int max, float *data) {
  std::mt19937_64 generator{1993764};
  std::uniform_int_distribution<> dist{min, max};
  for (int i = 0; i < size; ++i) {
    data[i] = dist(generator);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = atoi(argv[1]);

  // number of functions
  int N = 1999999;
  printf("N = %d\n", N);

  float *A, *B, *C, *D, *E;
  float *minimum_ref, *minimum;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  A = sycl::malloc_host<float>(N, q);
  B = sycl::malloc_host<float>(N, q);
  C = sycl::malloc_host<float>(N, q);
  D = sycl::malloc_host<float>(N, q);
  E = sycl::malloc_host<float>(N, q);
  minimum_ref = sycl::malloc_host<float>(N, q);
  minimum = sycl::malloc_host<float>(N, q);

  printf("generating data...\n");

  generate_data(N, -100, 100, A);
  generate_data(N, -100, 100, B);
  generate_data(N, -100, 100, C);
  generate_data(N, -100, 100, D);
  generate_data(N, -100, 100, E);

  for (int i = 0; i < N; i++) {
    if (A[i] == 0) {
      A[i] = 1;
    } // avoid undefined behaviour in solver when A=0
  }

  float dur = 0;
  float avg = 0;
  bool ok;

  printf("####################### Reference #############\n");

  for (int k = 0; k < repeat; ++k) {
    auto start = std::chrono::high_resolution_clock::now();

    QuarticMinimumCPU(N, A, B, C, D, E, minimum_ref);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    dur = elapsed.count() * 1000;
    // printf("Time (ms): %f\n", dur);
    avg += dur;
  }

  printf("Execution time (ms): %f\n", avg / repeat);

  avg = 0;

  printf("####################### GPU (no streams) #############\n");

  for (int k = 0; k < repeat; ++k) {

    auto start = std::chrono::high_resolution_clock::now();

    QuarticMinimumGPU(q, N, A, B, C, D, E, minimum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    dur = elapsed.count() * 1000;
    // printf("Time (ms): %f\n", dur);
    avg += dur;
  }

  printf("Execution time (ms): %f\n", avg / repeat);

  ok = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(minimum[i] - minimum_ref[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  avg = 0;

  printf("####################### GPU (streams) #############\n");

  for (int k = 0; k < repeat; ++k) {

    auto start = std::chrono::high_resolution_clock::now();

    QuarticMinimumGPUStreams(q, N, A, B, C, D, E, minimum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    dur = elapsed.count() * 1000;
    // printf("Time (ms): %f\n", dur);
    avg += dur;
  }

  printf("Execution time (ms): %f\n", avg / repeat);

  ok = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(minimum[i] - minimum_ref[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);
  sycl::free(D, q);
  sycl::free(E, q);
  sycl::free(minimum_ref, q);
  sycl::free(minimum, q);

  return 0;
}
