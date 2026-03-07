#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <cuda.h>
#include "benchmark.h"
#include "kernels.h"

void run_benchmark(const int repeat)
{
  int i, j, cnt, val_ref, val_eff;
  uint64_t time_vals[SIZES_CNT_MAX][BASES_CNT_MAX][2];

  uint32_t *d_bases32;
  uint32_t bases32_size = sizeof(bases32);
  cudaMalloc((void**)&d_bases32, bases32_size);
  cudaMemcpy(d_bases32, bases32, bases32_size, cudaMemcpyHostToDevice);

  uint32_t *d_n32;
  uint32_t n32_size = BENCHMARK_ITERATIONS * sizeof(uint32_t);
  cudaMalloc((void**)&d_n32, n32_size);

  int *d_val;
  int val_dev;
  cudaMalloc((void**)&d_val, sizeof(int));
  dim3 grids ((BENCHMARK_ITERATIONS + 255) / 256);
  dim3 blocks (256);

  printf("Starting benchmark...\n");

  bool ok = true;
  double mr32_sf_time = 0.0, mr32_eff_time = 0.0;

  for (i = 0; i < SIZES_CNT32; i++) {
    val_ref = val_eff = 0;
    cudaMemcpy(d_n32, n32[i], n32_size, cudaMemcpyHostToDevice);

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
      cudaMemset(d_val, 0, sizeof(int));
      cudaDeviceSynchronize();
      auto start = std::chrono::steady_clock::now();
      mr32_sf <<< grids, blocks >>> (d_bases32, d_n32, d_val, BENCHMARK_ITERATIONS);
      cudaDeviceSynchronize();
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    mr32_sf_time += time / repeat;

    cudaMemcpy(&val_dev, d_val, sizeof(int), cudaMemcpyDeviceToHost);
    if (val_ref != val_dev) {
      ok = false;
      fprintf(stderr, "Results mismatch: val_dev = %d, val_ref = %d\n", val_dev, val_ref);
      break;
    }

    time = 0.0;
    for (int n = 0; n < repeat; n++) {
      cudaMemset(d_val, 0, sizeof(int));
      cudaDeviceSynchronize();
      auto start = std::chrono::steady_clock::now();
      mr32_eff <<< grids, blocks >>> (d_bases32, d_n32, d_val, BENCHMARK_ITERATIONS);
      cudaDeviceSynchronize();
      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    mr32_eff_time += time / repeat;

    cudaMemcpy(&val_dev, d_val, sizeof(int), cudaMemcpyDeviceToHost);
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

  cudaFree(d_bases32);
  cudaFree(d_n32);
  cudaFree(d_val);
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
