/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "block_load.hpp"
#include "block_store.hpp"
#include "block_scan.hpp"

int grid_size = 1;
int repeat = 100;
bool g_verbose = false;

template <typename T>
int CompareDeviceResults(sycl::queue &q, T *h_reference, T *d_data, size_t num_items) {
  // Allocate array on host
  T *h_data = (T*) malloc(num_items * sizeof(T));

  // Copy data back
  q.memcpy(h_data, d_data, sizeof(T) * num_items).wait();

  // Check
  int retval = memcmp(h_data, h_reference, num_items * sizeof(T));

  // Cleanup
  if (h_data) free(h_data);

  return retval;
}

/**
 * Simple kernel for performing a block-wide exclusive prefix sum over integers
 */
template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockScanAlgorithm ALGORITHM>
void BlockPrefixSumKernel(const int *d_in,  // Tile of input
                                int *d_out, // Tile of output
                          const sycl::nd_item<3> &item)
{
  // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;

  // Specialize BlockStore type using warp-striped loads for coalescing,
  // then transposes in shared memory to a blocked arrangement)
  typedef BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;

  // Specialize BlockScan type for our thread block
  typedef BlockScan<int, BLOCK_THREADS, ALGORITHM> BlockScanT;

  // Shared memory
  union temp_storage
  {
    typename BlockLoadT::TempStorage    load;
    typename BlockStoreT::TempStorage   store;
    typename BlockScanT::TempStorage    scan;
  } ;

  auto &slm = *sycl::ext::oneapi::group_local_memory_for_overwrite<union temp_storage>(item.get_group());

  // Per-thread tile data
  int data[ITEMS_PER_THREAD];

  // Load items into a blocked arrangement
  BlockLoadT(slm.load, item).Load(d_in, data);

  // Barrier for smem reuse
  item.barrier(sycl::access::fence_space::local_space);

  // Compute exclusive prefix sum
  int aggregate;
  BlockScanT(slm.scan, item).ExclusiveSum(data, data, aggregate);

  // Barrier for smem reuse
  item.barrier(sycl::access::fence_space::local_space);

  // Store items from a blocked arrangement
  BlockStoreT(slm.store, item).Store(d_out, data);

  // Store aggregate and elapsed clocks
  if (item.get_local_id(2) == 0)
  {
    d_out[BLOCK_THREADS * ITEMS_PER_THREAD] = aggregate;
  }
}


/**
 * Initialize exclusive prefix sum problem (and solution).
 * Returns the aggregate
 */
int Initialize(
    int *h_in,
    int *h_reference,
    int num_items)
{
  int inclusive = 0;
  for (int i = 0; i < num_items; ++i)
  {
    h_in[i] = i % 17;
    h_reference[i] = inclusive;
    inclusive += h_in[i];
  }
  return inclusive;
}


/**
 * Test thread block scan
 */
template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockScanAlgorithm ALGORITHM>
void Test(sycl::queue &q) try {
  const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  // Allocate host arrays
  int *h_in        = new int[TILE_SIZE];
  int *h_reference = new int[TILE_SIZE];
  int *h_gpu       = new int[TILE_SIZE + 1];

  // Initialize problem and reference output on host
  int h_aggregate = Initialize(h_in, h_reference, TILE_SIZE);

  // Initialize device arrays
  int *d_in = sycl::malloc_device<int>(TILE_SIZE, q);
  int *d_out = sycl::malloc_device<int>((TILE_SIZE + 1), q);

  // Display input problem data
  if (g_verbose)
  {
    printf("Input data: ");
    for (int i = 0; i < TILE_SIZE; i++)
      printf("%d, ", h_in[i]);
    printf("\n\n");
  }

  // Copy problem to device
  q.memcpy(d_in, h_in, sizeof(int) * TILE_SIZE).wait();

  printf("BlockScan algorithm %s on %d items (%d timing iterations, %d blocks, %d threads, %d items per thread):\n",
      (ALGORITHM == BLOCK_SCAN_RAKING) ? "BLOCK_SCAN_RAKING" : (ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE) ? "BLOCK_SCAN_RAKING_MEMOIZE" : "BLOCK_SCAN_WARP_SCANS",
      TILE_SIZE, repeat, grid_size, BLOCK_THREADS, ITEMS_PER_THREAD);

  sycl::range<3> gws (1, 1, (size_t)grid_size * BLOCK_THREADS);
  sycl::range<3> lws (1, 1, BLOCK_THREADS);

  auto kFn = [=](sycl::nd_item<3> item) {
    BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>(
      d_in, d_out, item);
  };

  // Run aggregate/prefix kernel
  for (int i = 0; i < 100; ++i) {
    q.parallel_for(sycl::nd_range<3>(gws, lws), kFn);

    // Check results for the first warmup run
    if (i == 0) {
      printf("\tOutput items: ");
      int compare = CompareDeviceResults(q, h_reference, d_out, TILE_SIZE);
      printf("%s\n", compare ? "FAIL" : "PASS");

      // Check total aggregate
      printf("\tAggregate: ");
      compare = CompareDeviceResults(q, &h_aggregate, d_out + TILE_SIZE, 1);
      printf("%s\n", compare ? "FAIL" : "PASS");
    }
  }
  q.wait();

  // Run this several times and average the performance results
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i)
  {
    q.parallel_for(sycl::nd_range<3>(gws, lws), kFn);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  // Display timing results
  float avg_millis        = elapsed * 1e-6 / repeat;
  float avg_items_per_sec = float(TILE_SIZE) * grid_size / avg_millis / 1000.0f;

  printf("\tAverage kernel time: %.4f ms\n", avg_millis);
  printf("\tAverage million items / sec: %.4f\n", avg_items_per_sec);

  // Cleanup
  if (h_in) delete[] h_in;
  if (h_reference) delete[] h_reference;
  if (h_gpu) delete[] h_gpu;
  if (d_in) sycl::free(d_in, q);
  if (d_out) sycl::free(d_out, q);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("The benchmark evaluates the impacts of the number of threads ");
    printf("per block and the number of items per thread on the performance ");
    printf("of block-level scans\n");
    printf("Usage: %s <grid_size> <repeat>\n", argv[0]);
    printf("grid_size specifies the number of thread blocks");
    return 1;
  }

  grid_size = atoi(argv[1]);
  repeat = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Run tests
  Test<1024, 1, BLOCK_SCAN_RAKING>(q);
  Test<512, 2, BLOCK_SCAN_RAKING>(q);
  Test<256, 4, BLOCK_SCAN_RAKING>(q);
  Test<128, 8, BLOCK_SCAN_RAKING>(q);
  Test<64, 16, BLOCK_SCAN_RAKING>(q);
  Test<32, 32, BLOCK_SCAN_RAKING>(q);

  printf("-------------\n");

  Test<1024, 1, BLOCK_SCAN_RAKING_MEMOIZE>(q);
  Test<512, 2, BLOCK_SCAN_RAKING_MEMOIZE>(q);
  Test<256, 4, BLOCK_SCAN_RAKING_MEMOIZE>(q);
  Test<128, 8, BLOCK_SCAN_RAKING_MEMOIZE>(q);
  Test<64, 16, BLOCK_SCAN_RAKING_MEMOIZE>(q);
  Test<32, 32, BLOCK_SCAN_RAKING_MEMOIZE>(q);

  printf("-------------\n");

  Test<1024, 1, BLOCK_SCAN_WARP_SCANS>(q);
  Test<512, 2, BLOCK_SCAN_WARP_SCANS>(q);
  Test<256, 4, BLOCK_SCAN_WARP_SCANS>(q);
  Test<128, 8, BLOCK_SCAN_WARP_SCANS>(q);
  Test<64, 16, BLOCK_SCAN_WARP_SCANS>(q);
  Test<32, 32, BLOCK_SCAN_WARP_SCANS>(q);

  return 0;
}
