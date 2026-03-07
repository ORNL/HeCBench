#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <chrono>
#include "benchmark.h"
#include "kernels.h"

void run_benchmark(const int repeat)
{
  int i, j, cnt, val_ref, val_eff;
  uint64_t time_vals[SIZES_CNT_MAX][BASES_CNT_MAX][2];

  int bases32_size = sizeof(bases32) / sizeof(bases32[0]);

  const uint32_t *d_bases32 = bases32;
  uint32_t *d_n32 = (uint32_t*) malloc (sizeof(uint32_t) * BENCHMARK_ITERATIONS);
  int val_dev;
  int *d_val = &val_dev;

  printf("Starting benchmark...\n");

  bool ok = true;
  double mr32_sf_time = 0.0, mr32_eff_time = 0.0;

  #pragma omp target data map (to: d_bases32[0:bases32_size]) \
                          map (alloc: d_n32[0:BENCHMARK_ITERATIONS], d_val[0:1])
  {
    for (i = 0; i < SIZES_CNT32; i++) {
      val_ref = val_eff = 0;

      memcpy(d_n32, n32[i], sizeof(uint32_t) * BENCHMARK_ITERATIONS);
      #pragma omp target update to (d_n32[0:BENCHMARK_ITERATIONS])
      
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
        #pragma omp target 
        d_val[0] = 0;

        auto start = std::chrono::steady_clock::now();

        // the efficient version is faster than the simple version on a device
        mr32_sf(d_bases32, d_n32, d_val, BENCHMARK_ITERATIONS);

        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
      mr32_sf_time += time / repeat;

      #pragma omp target update from (d_val[0:1])

      if (val_ref != val_dev) {
        ok = false;
        fprintf(stderr, "Results mismatch: val_dev = %d, val_ref = %d\n", val_dev, val_ref);
        break;
      }

      time = 0.0;
      for (int n = 0; n < repeat; n++) {
        #pragma omp target 
        d_val[0] = 0;

        auto start = std::chrono::steady_clock::now();

        mr32_eff(d_bases32, d_n32, d_val, BENCHMARK_ITERATIONS);

        auto end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
      mr32_eff_time += time / repeat;

      #pragma omp target update from (d_val[0:1])

      if (val_ref != val_dev) {
        ok = false;
        fprintf(stderr, "Results mismatch: val_dev = %d, val_ref = %d\n", val_dev, val_ref);
        break;
      }
    }
  }

  printf("Total kernel execution time (mr32_simple  ): %f (ms)\n", mr32_sf_time * 1e-6);
  printf("Total kernel execution time (mr32_efficent): %f (ms)\n", mr32_eff_time * 1e-6);
  printf("%s\n", ok ? "PASS" : "FAIL");

  // device results are not included
  print_results(bits32, SIZES_CNT32, BASES_CNT32, time_vals);
  free(d_n32);
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

  printf("Setting random primes...\n");
  set_nprimes();
  run_benchmark(repeat);

  printf("Setting random odd integers...\n");
  set_nintegers();
  run_benchmark(repeat);

  if (!ok) return 1;
  return 0;
}
