#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sycl/sycl.hpp>

typedef unsigned int T;
typedef sycl::uint4 VECTYPE;

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
  for (int i = 0; i < size; i++)
  {
    h_idata[i] = i % 16; // Fill with some pattern
    h_odata[i] = size - i;
  }

  std::cout << "Running benchmark with input array length " << size << std::endl;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

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

  T *d_idata = sycl::malloc_device<T>(size, q);
  q.memcpy(d_idata, h_idata, size * sizeof(T));

  T *d_odata = sycl::malloc_device<T>(size, q);
  T *d_isums = sycl::malloc_device<T>(num_work_groups * num_digits, q);

  sycl::range<1> gws (global_wsize);
  sycl::range<1> lws (local_wsize);
  sycl::range<1> lws2 (local_wsize * 2);
  sycl::range<1> lws3 (16);

  double time = 0.0;

  for (int k = 0; k < passes; k++)
  {
    q.wait();
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

      auto in = even ? d_idata : d_odata;
      auto out = even ? d_odata : d_idata;
      auto isums = d_isums;

      q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor <T, 1> lmem (lws, cgh);
          cgh.parallel_for<class reduce>(
            sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
            #include "sort_reduce.sycl"
        });
      });

      q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor <T, 1> lmem (lws2, cgh);
        sycl::local_accessor <T, 0> l_seed (cgh);
        cgh.parallel_for<class top_scan>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          #include "sort_top_scan.sycl"
        });
      });

      q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor <T, 1> lmem (lws2, cgh);
        sycl::local_accessor <T, 1> l_scanned_seeds (lws3, cgh);
        sycl::local_accessor <T, 1> l_block_counts (lws3, cgh);
        cgh.parallel_for<class bottom_scan>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          #include "sort_bottom_scan.sycl"
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }  // passes

  printf("Average elapsed time of sort: %lf (s)\n", time * 1e-9 / passes);

  q.memcpy(h_odata, d_odata, size * sizeof(T)).wait();
  verifySort(h_odata, size);

  // reference sort
  time = 0.0;
  for (int k = 0; k < passes; k++) {
    q.memcpy(d_odata, h_idata, size * sizeof(T));
    q.wait();
    auto start = std::chrono::steady_clock::now();
    oneapi::dpl::sort(oneapi::dpl::execution::make_device_policy(q), d_odata, d_odata + size);
    q.wait();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average elapsed time of oneDPL::sort: %lf (s)\n", time * 1e-9 / passes);

  q.memcpy(h_odata, d_odata, size * sizeof(T)).wait();
  verifySort(h_odata, size);

  sycl::free(d_idata, q);
  sycl::free(d_odata, q);
  sycl::free(d_isums, q);

  free(h_idata);
  free(h_odata);
  return 0;
}
