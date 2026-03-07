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
#include <sycl/sycl.hpp>
#include "block.h"
#include <cstdlib>

//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------
struct CustomDifference
{
  template <typename DataType>
  DataType operator()(DataType &lhs, DataType &rhs)
  {
    return lhs - rhs;
  }
};

/**
 * Simple kernel for performing a block-wide adjacent difference.
 */
template <int BLOCK_THREADS>
void BlockAdjDiffKernel(const int* d_in, int* d_out, bool subtract_left,
                        const sycl::nd_item<3> &item)
{
  // Specialize BlockAdjacentDifference for a 1D block of threads of type int
   using BlockAdjacentDifferenceT = BlockAdjacentDifference<int, BLOCK_THREADS>;

   // Allocate shared memory for BlockAdjacentDifference
   auto g = item.get_group();
   sycl::multi_ptr<typename BlockAdjacentDifferenceT::TempStorage[1],
                   sycl::access::address_space::local_space> p1 =
      sycl::ext::oneapi::group_local_memory_for_overwrite<typename BlockAdjacentDifferenceT::TempStorage[1]>(g);
   auto &temp_storage = *p1;

   int idx = item.get_global_id(2);

   int thread_data[4];
   reinterpret_cast<sycl::int4 *>(thread_data)[0] =
       reinterpret_cast<const sycl::int4 *>(d_in)[idx];

   if (subtract_left) {
     BlockAdjacentDifferenceT(*temp_storage, item)
         .SubtractLeft(thread_data, thread_data, CustomDifference());
   }
   else {
     BlockAdjacentDifferenceT(*temp_storage, item)
         .SubtractRight(thread_data, thread_data, CustomDifference());
   }

   reinterpret_cast<sycl::int4 *>(d_out)[idx] =
       reinterpret_cast<sycl::int4 *>(thread_data)[0];
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
void Test(sycl::queue &q, int num_items, int repeat)
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
  d_in = sycl::malloc_device<int>(num_items, q);
  d_out = sycl::malloc_device<int>(num_items, q);

  int grid_size = num_items / items_per_block;

  sycl::range<3> gws (1, 1, grid_size * BLOCK_THREADS);
  sycl::range<3> lws (1, 1, BLOCK_THREADS);

  q.memcpy(d_in, h_in, sizeof(int) * num_items);
  for (int i = 0; i < repeat; i++) {
    q.parallel_for(
        sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
          BlockAdjDiffKernel<BLOCK_THREADS>(d_in, d_out, true, item);
    });
  }
  q.memcpy(h_out, d_out, sizeof(int) * num_items).wait();

  for (int b = 0; b < grid_size; b++) {
    auto in = h_in + b * items_per_block;
    auto out = r_out + b * items_per_block;
    for (int i = 0; i < items_per_block; i++) {
      out[i] = (i - 1) < 0 ? in[i] : in[i] - in[i-1];
    }
  }

  int compare = memcmp(r_out, h_out, sizeof(int) * num_items);
  printf("%s\n", compare ? "FAIL" : "PASS");

  q.memcpy(d_in, h_in, sizeof(int) * num_items);
  for (int i = 0; i < repeat; i++) {
    q.parallel_for(
        sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
          BlockAdjDiffKernel<BLOCK_THREADS>(d_in, d_out, false, item);
    });
  }
  q.memcpy(h_out, d_out, sizeof(int) * num_items).wait();

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
    q.parallel_for(
        sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
          BlockAdjDiffKernel<BLOCK_THREADS>(d_in, d_out, true, item);
    });
    q.parallel_for(
        sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
          BlockAdjDiffKernel<BLOCK_THREADS>(d_out, d_out, false, item);
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernels (thread block size = %4d): %f (us)\n",
         BLOCK_THREADS, (time * 1e-3f) / repeat);

  if (h_in) delete[] h_in;
  if (h_out) delete[] h_out;
  if (r_out) delete[] r_out;
  if (d_in) sycl::free(d_in, q);
  if (d_out) sycl::free(d_out, q);
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  Test<  64>(q, nelems, repeat);
  Test< 128>(q, nelems, repeat);
  Test< 256>(q, nelems, repeat);
  Test< 512>(q, nelems, repeat);
  Test<1024>(q, nelems, repeat);

  if (compare) return 1;
  if (compare) return 1;
  return 0;
}
