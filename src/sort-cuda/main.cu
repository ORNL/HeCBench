#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

typedef unsigned int T;
typedef uint4 VECTYPE;

// kernels
#include "sort_reduce.h"
#include "sort_top_scan.h"
#include "sort_bottom_scan.h"

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
  if (!passed) exit(1);
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

  T* h_idata = (T*) malloc (bytes); 
  T* h_odata = (T*) malloc (bytes); 

  // Initialize host memory
  std::cout << "Initializing host memory." << std::endl;
  for (unsigned int i = 0; i < size; i++)
  {
    h_idata[i] = i % 16; // Fill with some pattern
    h_odata[i] = size - i;
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

  T* d_idata;
  T* d_odata;
  T* d_isums;

  cudaMalloc((void**)&d_idata, size * sizeof(T));
  cudaMemcpyAsync(d_idata, h_idata, size * sizeof(T), cudaMemcpyHostToDevice, 0);
  cudaMalloc((void**)&d_odata, size * sizeof(T));
  cudaMalloc((void**)&d_isums, num_work_groups * num_digits * sizeof(T));

  T *d_in, *d_out;
  double time = 0.0;

  for (int k = 0; k < passes; k++)
  {
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // Assuming an 8 bit byte.
    // shift is uint because Computecpp compiler has no operator>>(unsigned int, int);
    for (unsigned int shift = 0; shift < sizeof(T)*8; shift += radix_width)
    {
      // Like scan, we use a reduce-then-scan approach
      // But before proceeding, update the shift appropriately
      // for each kernel. This is how many bits to shift to the
      // right used in binning.
      // Also, the sort is not in place, so swap the input and output
      // buffers on each pass.
      bool even = ((shift / radix_width) % 2 == 0) ? true : false;
      d_in = even ? d_idata : d_odata;
      d_out = even ? d_odata : d_idata;

      reduce<<<num_work_groups, local_wsize>>> (d_in, d_isums, size, shift);
      top_scan<<<1, local_wsize>>>(d_isums, num_work_groups);
      bottom_scan<<<num_work_groups, local_wsize>>>(d_out, d_in, d_isums, size, shift);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }  // passes

  printf("Average elapsed time of sort: %lf (s)\n", time * 1e-9 / passes);

  cudaMemcpy(h_odata, d_out, size * sizeof(T), cudaMemcpyDeviceToHost);
  verifySort(h_odata, size);

  // reference sort
  time = 0.0;
  for (int k = 0; k < passes; k++) {
    cudaMemcpy(d_odata, h_idata, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    thrust::device_ptr<T> d_out_ptr (d_odata);
    thrust::sort(d_out_ptr, d_out_ptr + size, thrust::less<T>());
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average elapsed time of Thrust::sort: %lf (s)\n", time * 1e-9 / passes);

  cudaMemcpy(h_odata, d_odata, size * sizeof(T), cudaMemcpyDeviceToHost);
  verifySort(h_odata, size);

  cudaFree(d_idata);
  cudaFree(d_odata);
  cudaFree(d_isums);

  free(h_idata);
  free(h_odata);
  return 0;
}
