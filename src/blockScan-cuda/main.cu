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
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cstdlib>

using namespace cub;

#define GPU_CHECK(x) do { \
  cudaError_t err = x; \
  if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while (0)

int grid_size = 1;
int repeat = 100;
bool g_verbose = false;

  template <typename T>
int CompareDeviceResults(T *h_reference, T *d_data, size_t num_items)
{
  // Allocate array on host
  T *h_data = (T*) malloc(num_items * sizeof(T));

  // Copy data back
  GPU_CHECK(cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost));

  // Check
  int retval = memcmp(h_data, h_reference, num_items * sizeof(T));

  // Cleanup
  if (h_data) free(h_data);

  return retval;
}

/**
 * Simple kernel for performing a block-wide exclusive prefix sum over integers
 */
template <int                     BLOCK_THREADS,
          int                     ITEMS_PER_THREAD,
          BlockScanAlgorithm      ALGORITHM>
__global__ void BlockPrefixSumKernel(
    const int   *d_in,          // Tile of input
    int         *d_out)         // Tile of output
{
  // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;

  // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
  typedef BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;

  // Specialize BlockScan type for our thread block
  typedef BlockScan<int, BLOCK_THREADS, ALGORITHM> BlockScanT;

  // Shared memory
  __shared__ union
  {
    typename BlockLoadT::TempStorage    load;
    typename BlockStoreT::TempStorage   store;
    typename BlockScanT::TempStorage    scan;
  } temp_storage;

  // Per-thread tile data
  int data[ITEMS_PER_THREAD];

  // Load items into a blocked arrangement
  BlockLoadT(temp_storage.load).Load(d_in, data);

  // Barrier for smem reuse
  __syncthreads();

  // Compute exclusive prefix sum
  int aggregate;
  BlockScanT(temp_storage.scan).ExclusiveSum(data, data, aggregate);

  // Barrier for smem reuse
  __syncthreads();

  // Store items from a blocked arrangement
  BlockStoreT(temp_storage.store).Store(d_out, data);

  // Store aggregate and elapsed clocks
  if (threadIdx.x == 0)
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
template <
int                 BLOCK_THREADS,
          int                 ITEMS_PER_THREAD,
  BlockScanAlgorithm  ALGORITHM>
void Test()
{
  const int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  // Allocate host arrays
  int *h_in        = new int[TILE_SIZE];
  int *h_reference = new int[TILE_SIZE];
  int *h_gpu       = new int[TILE_SIZE + 1];

  // Initialize problem and reference output on host
  int h_aggregate = Initialize(h_in, h_reference, TILE_SIZE);

  // Initialize device arrays
  int *d_in           = NULL;
  int *d_out          = NULL;
  GPU_CHECK(cudaMalloc((void**)&d_in, sizeof(int) * TILE_SIZE));
  GPU_CHECK(cudaMalloc((void**)&d_out, sizeof(int) * (TILE_SIZE + 1)));

  // Display input problem data
  if (g_verbose)
  {
    printf("Input data: ");
    for (int i = 0; i < TILE_SIZE; i++)
      printf("%d, ", h_in[i]);
    printf("\n\n");
  }

  // Copy problem to device
  GPU_CHECK(cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice));

  printf("BlockScan algorithm %s on %d items (%d timing iterations, %d blocks, %d threads, %d items per thread):\n",
      (ALGORITHM == BLOCK_SCAN_RAKING) ? "BLOCK_SCAN_RAKING" : (ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE) ? "BLOCK_SCAN_RAKING_MEMOIZE" : "BLOCK_SCAN_WARP_SCANS",
      TILE_SIZE, repeat, grid_size, BLOCK_THREADS, ITEMS_PER_THREAD);

  // Run aggregate/prefix kernel
  for (int i = 0; i < 100; ++i) {
    BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM><<<grid_size, BLOCK_THREADS>>>(
        d_in,
        d_out);

    // Check results for the first warmup run
    if (i == 0) {
      printf("\tOutput items: ");
      int compare = CompareDeviceResults(h_reference, d_out, TILE_SIZE);
      printf("%s\n", compare ? "FAIL" : "PASS");
      if (compare) exit(1);

      // Check total aggregate
      printf("\tAggregate: ");
      compare = CompareDeviceResults(&h_aggregate, d_out + TILE_SIZE, 1);
      printf("%s\n", compare ? "FAIL" : "PASS");
      if (compare) exit(1);
    }
  }
  GPU_CHECK(cudaDeviceSynchronize());

  // Run this several times and average the performance results
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i)
  {
    // Run aggregate/prefix kernel
    BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM><<<grid_size, BLOCK_THREADS>>>(
        d_in,
        d_out);
  }
  GPU_CHECK(cudaDeviceSynchronize());
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
  if (d_in) GPU_CHECK(cudaFree(d_in));
  if (d_out) GPU_CHECK(cudaFree(d_out));
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

  // Run tests
  Test<1024, 1, BLOCK_SCAN_RAKING>();
  Test<512, 2, BLOCK_SCAN_RAKING>();
  Test<256, 4, BLOCK_SCAN_RAKING>();
  Test<128, 8, BLOCK_SCAN_RAKING>();
  Test<64, 16, BLOCK_SCAN_RAKING>();
  Test<32, 32, BLOCK_SCAN_RAKING>();

  printf("-------------\n");

  Test<1024, 1, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<512, 2, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<256, 4, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<128, 8, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<64, 16, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<32, 32, BLOCK_SCAN_RAKING_MEMOIZE>();

  printf("-------------\n");

  Test<1024, 1, BLOCK_SCAN_WARP_SCANS>();
  Test<512, 2, BLOCK_SCAN_WARP_SCANS>();
  Test<256, 4, BLOCK_SCAN_WARP_SCANS>();
  Test<128, 8, BLOCK_SCAN_WARP_SCANS>();
  Test<64, 16, BLOCK_SCAN_WARP_SCANS>();
  Test<32, 32, BLOCK_SCAN_WARP_SCANS>();

  return 0;
}
