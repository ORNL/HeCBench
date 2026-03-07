/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <hipcub/hipcub.hpp>
#include <cstdlib>

//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------
struct CustomDifference
{
  template <typename DataType>
  __device__ DataType operator()(DataType &lhs, DataType &rhs)
  {
    return lhs - rhs;
  }
};

/**
 * Simple kernel for performing a block-wide adjacent difference.
 */
template <int BLOCK_THREADS>
__global__ void BlockAdjDiffKernel(const int* d_in, int* d_out, bool subtract_left)
{
  // Specialize BlockAdjacentDifference for a 1D block of threads of type int
   using BlockAdjacentDifferenceT = hipcub::BlockAdjacentDifference<int, BLOCK_THREADS>;

   // Allocate shared memory for BlockAdjacentDifference
   __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;

   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   int thread_data[4];
   reinterpret_cast<int4*>(thread_data)[0] = reinterpret_cast<const int4*>(d_in)[idx];

   if (subtract_left) {
     BlockAdjacentDifferenceT(temp_storage).SubtractLeft(
         thread_data,
         thread_data,
         CustomDifference());
   }
   else {
     BlockAdjacentDifferenceT(temp_storage).SubtractRight(
         thread_data,
         thread_data,
         CustomDifference());
   }

   reinterpret_cast<int4*>(d_out)[idx] = reinterpret_cast<int4*>(thread_data)[0];
}

//---------------------------------------------------------------------
// Host utilities
//---------------------------------------------------------------------

/**
 * Initialize reduction problem (and solution).
 */
void Initialize(int* h_in, int num_items)
{
  for (int i = 0; i < num_items; ++i)
  {
    h_in[i] = i % 17;
  }
}

template <int BLOCK_THREADS>
void Test(int num_items, int repeat)
{
  const int ITEMS_PER_THREAD = 4;
  const int items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  num_items = (num_items + items_per_block - 1) / items_per_block * items_per_block;

  int* h_in = new int[num_items];
  int* h_out = new int[num_items];
  int* r_out = new int[num_items];

  Initialize(h_in, num_items);

  int* d_in  = nullptr;
  int* d_out = nullptr;
  hipMalloc((void**) &d_in, sizeof(int) * num_items);
  hipMalloc((void**) &d_out, sizeof(int) * num_items);

  int grid_size = num_items / items_per_block;

  dim3 grid (grid_size);
  dim3 block (BLOCK_THREADS);

  // verify the SubtractLeft
  hipMemcpy(d_in, h_in, sizeof(int) * num_items, hipMemcpyHostToDevice);
  for (int i = 0; i < repeat; i++) {
    BlockAdjDiffKernel<BLOCK_THREADS><<<grid, block>>>(d_in, d_out, true);
  }
  hipMemcpy(h_out, d_out, sizeof(int) * num_items, hipMemcpyDeviceToHost);

  for (int b = 0; b < grid_size; b++) {
    auto in = h_in + b * items_per_block;
    auto out = r_out + b * items_per_block;
    for (int i = 0; i < items_per_block; i++) {
      out[i] = (i - 1) < 0 ? in[i] : in[i] - in[i-1];
    }
  }

  int compare = memcmp(r_out, h_out, sizeof(int) * num_items);
  printf("%s\n", compare ? "FAIL" : "PASS");

  // verify the SubtractRight
  hipMemcpy(d_in, h_in, sizeof(int) * num_items, hipMemcpyHostToDevice);
  for (int i = 0; i < repeat; i++) {
    BlockAdjDiffKernel<BLOCK_THREADS><<<grid, block>>>(d_in, d_out, false);
  }
  hipMemcpy(h_out, d_out, sizeof(int) * num_items, hipMemcpyDeviceToHost);

  for (int b = 0; b < grid_size; b++) {
    auto in = h_in + b * items_per_block;
    auto out = r_out + b * items_per_block;
    for (int i = 0; i < items_per_block; i++) {
      out[i] = (i + 1) >= items_per_block ? in[i] : in[i] - in[i+1];
    }
  }
  compare = memcmp(r_out, h_out, sizeof(int) * num_items);
  printf("%s\n", compare ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    BlockAdjDiffKernel<BLOCK_THREADS><<<grid, block>>>(d_in, d_out, true);
    BlockAdjDiffKernel<BLOCK_THREADS><<<grid, block>>>(d_out, d_out, false);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernels (thread block size = %4d): %f (us)\n",
         BLOCK_THREADS, (time * 1e-3f) / repeat);

  if (h_in) delete[] h_in;
  if (h_out) delete[] h_out;
  if (r_out) delete[] r_out;
  if (d_in) hipFree(d_in);
  if (d_out) hipFree(d_out);
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  Test<  64>(nelems, repeat);
  Test< 128>(nelems, repeat);
  Test< 256>(nelems, repeat);
  Test< 512>(nelems, repeat);
  Test<1024>(nelems, repeat);

  if (compare) return 1;
  if (compare) return 1;
  return 0;
}
