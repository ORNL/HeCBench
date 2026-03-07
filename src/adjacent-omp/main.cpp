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
#include <omp.h>
#include <cstdlib>

/**
 * Simple kernel for performing a block-wide adjacent difference.
 */
template <int BLOCK_THREADS>
void BlockAdjDiffKernel(const int* d_in, int* d_out, bool subtract_left, const int num_items)
{
   const int items_per_block = BLOCK_THREADS * 4;

   #pragma omp target teams distribute
   for (int b = 0; b < num_items / items_per_block; b++) {
     auto in = d_in + b * items_per_block;
     auto out = d_out + b * items_per_block;
     #pragma omp parallel for
     for (int i = 0; i < items_per_block; i++) {
       if (subtract_left)
         out[i] = (i - 1) < 0 ? in[i] : in[i] - in[i-1];
       else
         out[i] = (i + 1) >= items_per_block ? in[i] : in[i] - in[i+1];
     }
   }
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
  const int grid_size = num_items / items_per_block;

  int* h_in = new int[num_items];
  int* h_out = new int[num_items];
  int* r_out = new int[num_items];

  Initialize(h_in, num_items);

  #pragma omp target data map(to: h_in[0:num_items]) \
                          map(alloc: h_out[0:num_items])
  {
    for (int i = 0; i < repeat; i++) {
      BlockAdjDiffKernel<BLOCK_THREADS>(h_in, h_out, true, num_items);
    }
    #pragma omp target update from (h_out[0:num_items])

    for (int b = 0; b < grid_size; b++) {
      auto in = h_in + b * items_per_block;
      auto out = r_out + b * items_per_block;
      for (int i = 0; i < items_per_block; i++) {
        out[i] = (i - 1) < 0 ? in[i] : in[i] - in[i-1];
      }
    }

    int compare = memcmp(r_out, h_out, sizeof(int) * num_items);
    printf("%s\n", compare ? "FAIL" : "PASS");
    if (compare) exit(1);

    for (int i = 0; i < repeat; i++) {
      BlockAdjDiffKernel<BLOCK_THREADS>(h_in, h_out, false, num_items);
    }
    #pragma omp target update from (h_out[0:num_items])

    for (int b = 0; b < grid_size; b++) {
      auto in = h_in + b * items_per_block;
      auto out = r_out + b * items_per_block;
      for (int i = 0; i < items_per_block; i++) {
        out[i] = (i + 1) >= items_per_block ? in[i] : in[i] - in[i+1];
      }
    }

    compare = memcmp(r_out, h_out, sizeof(int) * num_items);
    printf("%s\n", compare ? "FAIL" : "PASS");
    if (compare) exit(1);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      BlockAdjDiffKernel<BLOCK_THREADS>(h_in, h_out, true, num_items);
      BlockAdjDiffKernel<BLOCK_THREADS>(h_out, h_out, false, num_items);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of the kernels (thread block size = %4d): %f (us)\n",
           BLOCK_THREADS, (time * 1e-3f) / repeat);
  }
  if (h_in) delete[] h_in;
  if (h_out) delete[] h_out;
  if (r_out) delete[] r_out;
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

  return 0;
}
