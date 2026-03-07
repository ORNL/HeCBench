#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "benchmark.h"
#include "kernels.h"

void run_benchmark(sycl::queue &q, const int repeat)
{
  int i, j, cnt, val_ref, val_eff;
  uint64_t time_vals[SIZES_CNT_MAX][BASES_CNT_MAX][2];

  uint32_t bases32_size = sizeof(bases32);
  uint32_t *d_bases32 = (uint32_t*) sycl::malloc_device(bases32_size, q);
  q.memcpy(d_bases32, bases32, bases32_size);

  uint32_t n32_size = BENCHMARK_ITERATIONS * sizeof(uint32_t);
  uint32_t *d_n32 = sycl::malloc_device<uint32_t>(BENCHMARK_ITERATIONS, q);

  int val_dev;
  int *d_val = sycl::malloc_device<int>(1, q);

  sycl::range<1> gws ((BENCHMARK_ITERATIONS + 255) / 256 * 256);
  sycl::range<1> lws (256);

  printf("Starting benchmark...\n");

  bool ok = true;
  double mr32_sf_time = 0.0, mr32_eff_time = 0.0;

  for (i = 0; i < SIZES_CNT32; i++) {
    val_ref = val_eff = 0;

    q.memcpy(d_n32, n32[i], n32_size);

    for (cnt = 1; cnt <= BASES_CNT32; cnt++) {
      time_point start = get_time();
      for (j = 0; j < BENCHMARK_ITERATIONS; j++)
        val_eff += efficient_mr32(bases32, cnt, n32[i][j]);
      time_vals[i][cnt - 1][0] = elapsed_time(start);
    }

    for (cnt = 1; cnt <= BASES_CNT32; cnt++) {
      time_point start = get_time();
      for (j = 0; j < BENCHMARK_ITERATIONS; j++)
        val_ref += straightforward_mr32(bases32, cnt, n32[i][j]);
      time_vals[i][cnt - 1][1] = elapsed_time(start);
    }

    // verify the results of simple and efficient versions on a host
    if (val_ref != val_eff) {
      ok = false;
      fprintf(stderr, "Results mismatch: val_ref = %d, val_eff = %d\n", val_ref, val_eff);
      break;
    }

    double time = 0.0;
    for (int n = 0; n < repeat; n++) { 
      q.memset(d_val, 0, sizeof(int)).wait();
      auto start = std::chrono::steady_clock::now();
      // the efficient version is faster than the simple version on a device
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class simple>(
         sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
         mr32_sf(item, d_bases32, d_n32, d_val, BENCHMARK_ITERATIONS);
        });
      }).wait();
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    mr32_sf_time += time / repeat;

    q.memcpy(&val_dev, d_val, sizeof(int)).wait();

    if (val_ref != val_dev) {
      ok = false;
      fprintf(stderr, "Results mismatch: val_dev = %d, val_ref = %d\n", val_dev, val_ref);
      break;
    }

    time = 0.0;
    for (int n = 0; n < repeat; n++) { 
      q.memset(d_val, 0, sizeof(int)).wait();
      auto start = std::chrono::steady_clock::now();
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class efficient>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          mr32_eff(item, d_bases32, d_n32, d_val, BENCHMARK_ITERATIONS);
        });
      }).wait();
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    mr32_eff_time += time / repeat;

    q.memcpy(&val_dev, d_val, sizeof(int)).wait();

    if (val_ref != val_dev) {
      ok = false;
      fprintf(stderr, "Results mismatch: val_dev = %d, val_ref = %d\n", val_dev, val_ref);
      break;
    }
  }

  printf("Total kernel execution time (mr32_simple  ): %f (ms)\n", mr32_sf_time * 1e-6);
  printf("Total kernel execution time (mr32_efficent): %f (ms)\n", mr32_eff_time * 1e-6);
  printf("%s\n", ok ? "PASS" : "FAIL");

  // device results are not included
  print_results(bits32, SIZES_CNT32, BASES_CNT32, time_vals);

  sycl::free(d_bases32, q);
  sycl::free(d_n32, q);
  sycl::free(d_val, q);
}

int main(int argc, char *argv[]) 
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  if (repeat <= 0) return 1;

#ifdef _WIN32
  system("mode CON: COLS=98");
#endif

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("Setting random primes...\n");
  set_nprimes();
  run_benchmark(q, repeat);

  printf("Setting random odd integers...\n");
  set_nintegers();
  run_benchmark(q, repeat);

  if (!ok) return 1;
  return 0;
}
