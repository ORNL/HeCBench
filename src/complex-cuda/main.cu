#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include "complex.h"
#include "kernels.h"
#include "reference.h"

bool check (const char *cs, int n)
{
  bool ok = true;
  for (int i = 0; i < n; i++) {
    if (cs[i] != 5) {
      ok = false; 
      break;
    }
  }
  return ok;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <problem size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  char* cs = (char*) malloc (n);

  char* d_cs;
  cudaMalloc((void**)&d_cs, n);

  dim3 grids ((n + 255)/256); 
  dim3 blocks (256);

  // warmup
  complex_float<<<grids, blocks>>>(d_cs, n);
  complex_double<<<grids, blocks>>>(d_cs, n);
  cudaDeviceSynchronize();

  printf("\nSingle-precision complex data type\n");
  auto start = std::chrono::steady_clock::now();

  // complex numbers in single precision
  for (int i = 0; i < repeat; i++) {
    complex_float<<<grids, blocks>>>(d_cs, n);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", time * 1e-9f / repeat);

  cudaMemcpy(cs, d_cs, n, cudaMemcpyDeviceToHost);
  bool complex_float_check = check(cs, n);

  start = std::chrono::steady_clock::now();

  // complex numbers in single precision
  for (int i = 0; i < repeat; i++) {
    ref_complex_float<<<grids, blocks>>>(d_cs, n);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (reference) %f (s)\n", time * 1e-9f / repeat);

  cudaMemcpy(cs, d_cs, n, cudaMemcpyDeviceToHost);
  complex_float_check &= check(cs, n);

  printf("\nDouble-precision complex data type\n");
  start = std::chrono::steady_clock::now();

  // complex numbers in double precision
  for (int i = 0; i < repeat; i++) {
    complex_double<<<grids, blocks>>>(d_cs, n);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", time * 1e-9f / repeat);

  cudaMemcpy(cs, d_cs, n, cudaMemcpyDeviceToHost);
  bool complex_double_check = check(cs, n);

  start = std::chrono::steady_clock::now();

  // complex numbers in double precision
  for (int i = 0; i < repeat; i++) {
    ref_complex_double<<<grids, blocks>>>(d_cs, n);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (reference) %f (s)\n", time * 1e-9f / repeat);

  cudaMemcpy(cs, d_cs, n, cudaMemcpyDeviceToHost);
  complex_double_check &= check(cs, n);

  printf("%s\n", (complex_float_check && complex_double_check)
                 ? "PASS" : "FAIL");
  if (!(complex_float_check && complex_double_check)) exit(1);

  cudaFree(d_cs);
  free(cs);

  return 0;
}
