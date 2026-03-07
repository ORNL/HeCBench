// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include "kernels.h"
#include "reference.h"
#include <cstdlib>

void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

static void bit_rev(fr_t* d_out, const fr_t* d_inp, uint32_t lg_domain_size)
{
  size_t domain_size = (size_t)1 << lg_domain_size;
  const uint32_t Z_COUNT = 256 / sizeof(fr_t); // 4: 64, 8: 32
  const uint32_t bsize = Z_COUNT>WARP_SZ ? Z_COUNT : WARP_SZ;

  if (domain_size <= 1024)
    bit_rev_permutation<<<1, domain_size>>>
      (d_out, d_inp, lg_domain_size);

  else if (domain_size < bsize * Z_COUNT)
    bit_rev_permutation<<<domain_size / WARP_SZ, WARP_SZ>>>
      (d_out, d_inp, lg_domain_size);

  else if (Z_COUNT > WARP_SZ || lg_domain_size <= 32)
    bit_rev_permutation_z<<<domain_size / Z_COUNT / bsize, bsize,
      bsize * Z_COUNT * sizeof(fr_t)>>>
        (d_out, d_inp, lg_domain_size);
  else {
    // Those GPUs that can reserve 96KB of shared memory can
    // schedule 2 blocks to each SM...
    int numProcs;
    cudaDeviceGetAttribute(&numProcs, cudaDevAttrMultiProcessorCount, 0);
    bit_rev_permutation_z<<<numProcs*2, BLOCK_SIZE, BLOCK_SIZE * Z_COUNT * sizeof(fr_t)>>>
        (d_out, d_inp, lg_domain_size);
  }
}

void bit_permute(const int lg_domain_size, const int repeat)
{
  assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
  size_t domain_size = (size_t)1 << lg_domain_size;
  size_t domain_size_bytes = sizeof(fr_t) *  domain_size;

  printf("Domain size is %zu\n", domain_size);

  fr_t *d_inout;
  cudaCheck(cudaMalloc(&d_inout, domain_size_bytes));
  fr_t *inout = (fr_t*) malloc (domain_size_bytes);
  fr_t *out = (fr_t*) malloc (domain_size_bytes);
  #pragma omp parallel for
  for (size_t i = 0; i < domain_size; i++) {
    out[i] = inout[i] = i;
  }

  // warmup and verify
  cudaCheck(cudaMemcpy(d_inout, inout, domain_size_bytes, cudaMemcpyHostToDevice));
  bit_rev(d_inout, d_inout, lg_domain_size);
  bit_rev_cpu(out, inout, lg_domain_size);
  cudaCheck(cudaMemcpy(inout, d_inout, domain_size_bytes, cudaMemcpyDeviceToHost));
  int error = memcmp(out, inout, domain_size_bytes);
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    bit_rev(d_inout, d_inout, lg_domain_size);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n\n", (time * 1e-3f) / repeat);

  cudaCheck(cudaFree(d_inout));
  free(inout);
  free(out);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  bit_permute(10, repeat);
  bit_permute(11, repeat);
  bit_permute(15, repeat);
  bit_permute(27, repeat);
  bit_permute(28, repeat);
}
