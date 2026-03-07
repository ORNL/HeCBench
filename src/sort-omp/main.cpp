#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

typedef unsigned int T;

template<typename T>
struct vec4 {
  T x;
  T y;
  T z;
  T w;
};

void verifySort(const T *keys, const size_t size)
{
  bool passed = true;
  for (size_t i = 0; i < size - 1; i++)
  {
    if (keys[i] > keys[i + 1])
    {
      passed = false;
#ifdef VERBOSE_OUTPUT
      std::cout << "Idx: " << i;
      std::cout << " Key: " << keys[i] << "\n";
#endif
      break;
    }
  }
  if (passed)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
}

int main(int argc, char** argv) 
{
  if (argc != 3) 
  {
    printf("Usage: %s <problem size> <number of passes>\n.", argv[0]);
    return -1;
  }

  int select = atoi(argv[1]);
  int passes = atoi(argv[2]);

  // Problem Sizes
  int probSizes[4] = { 1, 8, 32, 64 };
  size_t size = probSizes[select];

  // Convert to MiB
  size = (size * 1024 * 1024) / sizeof(T);

  // Create input data on CPU
  unsigned int bytes = size * sizeof(T);

  T* idata = (T*) malloc (bytes); 
  T* odata = (T*) malloc (bytes); 

  // Initialize host memory
  std::cout << "Initializing host memory." << std::endl;
  for (int i = 0; i < size; i++)
  {
    idata[i] = i % 16; // Fill with some pattern
    odata[i] = size - i;
  }

  std::cout << "Running benchmark with input array length " << size << std::endl;

  // Number of local work items per group
  const size_t local_wsize  = 256;
  // Number of global work items
  const size_t global_wsize = 16384; 
  // 64 work groups
  const size_t num_work_groups = global_wsize / local_wsize;

  // The radix width in bits
  const int radix_width = 4; // Changing this requires major kernel updates
  //const int num_digits = (int)pow((double)2, radix_width); // n possible digits
  const int num_digits = 16;

  T* isums = (T*) malloc (sizeof(T) * num_work_groups * num_digits);

  #pragma omp target data map(to: idata[0:size]) \
                          map(from: odata[0:size]) \
                          map(alloc: isums[0:num_work_groups * num_digits])
  {
    double time = 0.0;

    for (int k = 0; k < passes; k++)
    {
      auto start = std::chrono::steady_clock::now();

      // Assuming an 8 bit byte.
      for (unsigned int shift = 0; shift < sizeof(T)*8; shift += radix_width)
      {
        // Like scan, we use a reduce-then-scan approach

        // But before proceeding, update the shift appropriately
        // for each kernel. This is how many bits to shift to the
        // right used in binning.

        // Also, the sort is not in place, so swap the input and output
        // buffers on each pass.
        bool even = ((shift / radix_width) % 2 == 0) ? true : false;

        T *in = even ? idata : odata;
        T *out = even ? odata : idata;

        #pragma omp target teams num_teams(num_work_groups) thread_limit(local_wsize)
        {
          T lmem[local_wsize];
          #pragma omp parallel
          {
            #include "sort_reduce.h"
          }
        }

#ifdef DEBUG
#pragma omp target update from (isums[0:num_work_groups * num_digits])
        for (int i = 0; i < num_work_groups * num_digits; i++)
          printf("reduce: %d: %d\n", shift, isums[i]);
#endif

        #pragma omp target teams num_teams(num_work_groups) thread_limit(local_wsize)
        {
          T lmem[local_wsize*2];
          T s_seed;
          #pragma omp parallel
          {
            #include "sort_top_scan.h"
          }
        }

#ifdef DEBUG
#pragma omp target update from (isums[0:num_work_groups * num_digits])
        for (int i = 0; i < num_work_groups * num_digits; i++)
          printf("top-scan: %d: %d\n", shift, isums[i]);
#endif

        #pragma omp target teams num_teams(num_work_groups) thread_limit(local_wsize)
        {
          T lmem[local_wsize*2];
          T l_scanned_seeds[16];
          T l_block_counts[16];
          #pragma omp parallel
          {
            #include "sort_bottom_scan.h"
          }
        }
      }

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }  // passes

    printf("Average elapsed time per pass %lf (s)\n", time * 1e-9 / passes);
  }

  verifySort(odata, size);

  free(idata);
  free(isums);
  free(odata);
  if (!passed) return 1;
  return 0;
}
