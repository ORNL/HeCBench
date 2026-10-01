//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Bitonic Sort: this algorithm converts a randomized sequence of numbers into
// a bitonic sequence (two ordered sequences), and then merge these two ordered
// sequences into a ordered sequence. Bitonic sort algorithm is briefly
// described as followed:
//
// - First, it decomposes the randomized sequence of size 2**n into 2**(n-1)
// pairs where each pair consists of 2 consecutive elements. Note that each pair
// is a bitonic sequence.
// - Step 0: for each pair (sequence of size 2), the two elements are swapped so
// that the two consecutive pairs form  a bitonic sequence in increasing order,
// the next two pairs form the second bitonic sequence in decreasing order, the
// next two pairs form the third bitonic sequence in  increasing order, etc, ...
// . At the end of this step, we have 2**(n-1) bitonic sequences of size 2, and
// they follow an order increasing, decreasing, increasing, .., decreasing.
// Thus, they form 2**(n-2) bitonic sequences of size 4.
// - Step 1: for each new 2**(n-2) bitonic sequences of size 4, (each new
// sequence consists of 2 consecutive previous sequences), it swaps the elements
// so that at the end of step 1, we have 2**(n-2) bitonic sequences of size 4,
// and they follow an order: increasing, decreasing, increasing, ...,
// decreasing. Thus, they form 2**(n-3) bitonic sequences of size 8.
// - Same logic applies until we reach the last step.
// - Step n: at this last step, we have one bitonic sequence of size 2**n. The
// elements in the sequence are swapped until we have a sequence in increasing
// order.
//
// In this implementation, a randomized sequence of size 2**n is given (n is a
// positive number). Compare-exchange partners use the classic XOR mapping.
// Stages whose partner distance is at least the team size run in global
// memory. Remaining intra-team stages are fused into a local-memory kernel
// to cut launch overhead.
//
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <omp.h>

#define BLOCK_SIZE 256

void ParallelBitonicSort(int input[], int n) {

  int64_t size = (int64_t)1 << n;
  const int64_t grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  #pragma omp target data map(tofrom: input[0:size])
  {
    auto start = std::chrono::steady_clock::now();

    for (int64_t k = 2; k <= size; k <<= 1) {
      for (int64_t j = k >> 1; j > 0; j >>= 1) {
        if (j >= BLOCK_SIZE) {
          #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
          for (int64_t i = 0; i < size; i++) {
            int64_t ixj = i ^ j;
            if (ixj > i) {
              bool increasing = ((i & k) == 0);
              int ai = input[i];
              int aj = input[ixj];
              if (increasing ? (ai > aj) : (ai < aj)) {
                input[i] = aj;
                input[ixj] = ai;
              }
            }
          }
        } else {
          #pragma omp target teams num_teams(grid) thread_limit(BLOCK_SIZE)
          {
            int s[BLOCK_SIZE];
            #pragma omp parallel num_threads(BLOCK_SIZE)
            {
              // The device may create fewer teams than requested; stride over
              // tiles the way the CUDA/HIP/SYCL kernels do.
              const int64_t tid = omp_get_thread_num();
              const int64_t nthreads = omp_get_num_threads();
              const int64_t stride =
                  (int64_t)omp_get_num_teams() * BLOCK_SIZE;

              for (int64_t tile = (int64_t)omp_get_team_num() * BLOCK_SIZE;
                   tile < size; tile += stride) {
                for (int64_t tx = tid; tx < BLOCK_SIZE; tx += nthreads) {
                  int64_t i = tile + tx;
                  s[tx] = (i < size) ? input[i] : 0;
                }
                #pragma omp barrier

                for (int64_t jj = j; jj > 0; jj >>= 1) {
                  for (int64_t tx = tid; tx < BLOCK_SIZE; tx += nthreads) {
                    int64_t i = tile + tx;
                    int64_t ixj = tx ^ jj;
                    if (i < size && ixj > tx) {
                      bool increasing = ((i & k) == 0);
                      int ai = s[tx];
                      int aj = s[ixj];
                      if (increasing ? (ai > aj) : (ai < aj)) {
                        s[tx] = aj;
                        s[ixj] = ai;
                      }
                    }
                  }
                  #pragma omp barrier
                }

                for (int64_t tx = tid; tx < BLOCK_SIZE; tx += nthreads) {
                  int64_t i = tile + tx;
                  if (i < size)
                    input[i] = s[tx];
                }
                #pragma omp barrier
              }
            }
          }
          break;
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time: %f (ms)\n", time * 1e-6f);
  }
}

void Usage(std::string prog_name, int exponent) {
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: " << prog_name << " n k \n\n";
  std::cout << " n: Integer exponent presenting the size of the input array. "
               "The number of element in\n";
  std::cout << "    the array must be power of 2 (e.g., 1, 2, 4, ...). Please "
               "enter the corresponding\n";
  std::cout << "    exponent between 0 and " << exponent - 1 << ".\n";
  std::cout << " k: Seed used to generate a random sequence.\n";
}

int main(int argc, char *argv[]) {
  int n, seed;
  int64_t size;
  // n must keep 2^n in int64_t and 2^n * sizeof(int) in size_t.
  int exp_max = (int)sizeof(size_t) * 8 - 2;
  if (exp_max > std::numeric_limits<int64_t>::digits)
    exp_max = std::numeric_limits<int64_t>::digits;

  // Read parameters.
  try {
    n = std::stoi(argv[1]);

    // Verify the boundary of acceptance.
    if (n < 0 || n >= exp_max) {
      Usage(argv[0], exp_max);
      return -1;
    }

    seed = std::stoi(argv[2]);
    size = (int64_t)1 << n;
  } catch (...) {
    Usage(argv[0], exp_max);
    return -1;
  }

  std::cout << "\nArray size: " << size << ", seed: " << seed << "\n";

  size_t size_bytes = (size_t)size * sizeof(int);

  // Memory allocated for host access only.
  int *data_cpu = (int *)malloc(size_bytes);

  // Memory allocated to store gpu results
  int *data_gpu = (int *)malloc(size_bytes);

  // Initialize the array randomly using a seed.
  srand(seed);

  for (int64_t i = 0; i < size; i++) {
    data_gpu[i] = data_cpu[i] = rand() % 1000;
  }

  std::cout << "Bitonic sort (parallel)..\n";
  ParallelBitonicSort(data_gpu, n);

  std::cout << "Reference sort (std::sort)..\n";
  std::sort(data_cpu, data_cpu + size);

  // Verify
  int unequal = memcmp(data_gpu, data_cpu, size_bytes);
  std::cout << (unequal ? "FAIL" : "PASS") << std::endl;

  // Clean CPU memory.
  free(data_cpu);
  free(data_gpu);

  return 0;
}
