/*
MPC code [double] (LnVs BIT LVs ZE): A GPU-based compressor for arrays of 
double-precision floating-point values.  See the following publication for
more information: http://cs.txstate.edu/~mb92/papers/cluster15.pdf.

Copyright (c) 2015-2020, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Annie Yang and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/MPC/.

Publication: This work is described in detail in the following paper.
Annie Yang, Hari Mukka, Farbod Hesaaraki, and Martin Burtscher. MPC: A
Massively Parallel Compression Algorithm for Scientific Data. Proceedings
of the IEEE International Conference on Cluster Computing, pp. 381-389.
September 2015.
*/

#include <cstdio>
#include <cassert>
#include <string>
#include <chrono>
#include <sycl/sycl.hpp>
#include "utils.h"

using std::string;

#define TPB 1024  /* do not change */

template <int WarpSize>
static inline 
void prefixsum(int &val, int sbuf[TPB], sycl::nd_item<1> &item)
{
  const int warp = item.get_local_id(0) / WarpSize;
  const int lane = item.get_local_id(0) & (WarpSize - 1);

  auto sg = item.get_sub_group();

  for (int d = 1; d < WarpSize; d *= 2) {
    int tmp = sycl::shift_group_right(sg, val, d);
    if (lane >= d) val += tmp;
  }
  if (lane == WarpSize - 1) sbuf[warp] = val;

  item.barrier(sycl::access::fence_space::local_space);

  if (warp == 0) {
    int v = sbuf[lane];
    for (int d = 1; d < WarpSize; d *= 2) {
      int tmp = sycl::shift_group_right(sg, v, d);
      if (lane >= d) v += tmp;
    }
    sbuf[lane] = v;
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (warp > 0) {
    val += sbuf[warp - 1];
  }
}

template <int WarpSize>
static inline 
void prefixsumlong(long &val, long sbuf[TPB], sycl::nd_item<1> &item)
{
  const int warp = item.get_local_id(0) / WarpSize;
  const int lane = item.get_local_id(0) & (WarpSize - 1);

  auto sg = item.get_sub_group();

  for (int d = 1; d < WarpSize; d *= 2) {
    unsigned int tmpl = sycl::shift_group_right(sg, (int)val, d);
    long tmph = sycl::shift_group_right(sg, (int)(val >> 32), d);
    if (lane >= d) val += (tmph << 32) + tmpl;
  }
  if (lane == WarpSize - 1) sbuf[warp] = val;

  item.barrier(sycl::access::fence_space::local_space);

  if (warp == 0) {
    long v = sbuf[lane];
    for (int d = 1; d < 32; d *= 2) {
      unsigned int tmpl = sycl::shift_group_right(sg, (int)v, d);
      long tmph = sycl::shift_group_right(sg, (int)(v >> 32), d);
      if (lane >= d) v += (tmph << 32) + tmpl;
    }
    sbuf[lane] = v;
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (warp > 0) {
    val += sbuf[warp - 1];
  }
}

template <int WarpSize>
static inline 
void prefixsumdimlong(long &val, long sbuf[TPB], const unsigned char dim,
                      sycl::nd_item<1> &item)
{
  const int tid = item.get_local_id(0);
  const int warp = tid / WarpSize;
  const int lane = tid & (WarpSize - 1);
  const int tix = (warp * dim) + (tid % dim);

  auto sg = item.get_sub_group();

  for (int d = dim; d < WarpSize; d *= 2) {
    unsigned int tmpl = sycl::shift_group_right(sg, (int)val, d);
    long tmph = sycl::shift_group_right(sg, (int)(val >> 32), d);
    if (lane >= d) val += (tmph << 32) + tmpl;
  }
  if ((lane + dim) > (WarpSize - 1)) sbuf[tix] = val;

  item.barrier(sycl::access::fence_space::local_space);

  if (warp < dim) {
    const int idx = (lane * dim) + warp;
    long v = sbuf[idx];
    for (int d = 1; d < WarpSize; d *= 2) {
      unsigned int tmpl = sycl::shift_group_right(sg, (int)v, d);
      long tmph = sycl::shift_group_right(sg, (int)(v >> 32), d);
      if (lane >= d) v += (tmph << 32) + tmpl;
    }
    sbuf[idx] = v;
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (warp > 0) {
    val += sbuf[tix - dim];
  }
}

/*****************************************************************************
This is the GPU compression kernel, which requires 1024 threads per block and
should be launched with as many blocks as the GPU can run simultaneously.

Inputs
------
n: the number of double values to be compressed
original: the input array holding the n doubles (has to be cast to a long array)
goffset: a temporary array with m elements where m = number of thread blocks
dim: the dimensionality of the input data (dim must be between 1 and 32)

Output
------
compressed: the output array that holds the compressed data in integer format

The output array needs to provide space for up to 2 + n + (n + 31) / 32 elements.
The upper half of the first element specifies how many elements are actually
used.  It should be replaced by the value n before the data is further processed.
*****************************************************************************/

template <int WarpSize>
static 
void MPCcompress(
  const int n, 
  long* __restrict const original,
  long* __restrict const compressed,
  volatile int* __restrict const goffset,
  unsigned char dim,
  sycl::nd_item<1> &item,
  int &start,
  int &top,
  long *sbuf1,
  long *sbuf2)
{
  const int tid = item.get_local_id(0);
  const int tidm1 = tid - 1;
  const int tidmdim = tid - dim;
  const int lanex = tid & 63;
  const int warpx = tid & 0x3c0;
  const int bid = item.get_group(0);
  const int gdim = item.get_group_range(0);
  const int bid1 = ((bid + 1) == gdim) ? 0 : (bid + 1);
  const int init = 1 + (n + 63) / 64;
  const int chunksm1 = ((n + (TPB - 1)) / TPB) - 1;
  auto sg = item.get_sub_group();

  for (int chunk = bid; chunk <= chunksm1; chunk += gdim) {
    const int idx = tid + chunk * TPB;

    long v1 = 0;
    if (idx < n) {
      v1 = original[idx];
      sbuf1[tid] = v1;
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (tid >= dim) {
      if (idx < n) {
        v1 -= sbuf1[tidmdim];
      }
    }
    sbuf2[tid] = v1;

    item.barrier(sycl::access::fence_space::local_space);

    long v2 = 0;

    for (int i = 63; i >= 0; i--) {
      v2 = (v2 << 1) + ((sbuf2[warpx + i] >> lanex) & 1);
    }
    sbuf1[tid] = v2;

    item.barrier(sycl::access::fence_space::local_space);

    if (tid > 0) {
      v2 -= sbuf1[tidm1];
    }

    int loc = 0;
    if (v2 != 0) loc = 1;

    unsigned long long bitmap = sycl::reduce_over_group(
        sg, loc ? (1UL << sg.get_local_linear_id()) : 0,
        sycl::ext::oneapi::plus<>());

    if constexpr (WarpSize == 64) {
      if (idx < n) compressed[1 + idx / 64] = bitmap;
    } else {
      if (lanex == 32) {
        sbuf2[tid] = bitmap;
      }

      item.barrier(sycl::access::fence_space::local_space);

      if (lanex == 0) {
        if (idx < n) compressed[1 + idx / 64] = (sbuf2[tid + 32] << 32) + bitmap;
      }
    }

    prefixsum<WarpSize>(loc, (int *)sbuf1, item);

    if (v2 != 0) {
      sbuf2[loc - 1] = v2;
    }

    if (tid == (TPB - 1)) {
      int st = init;
      if (chunk > 0) {
        do {
          st = goffset[bid];
        } while (st < 0);  // busy waiting
      }
      goffset[bid1] = st + loc;
      goffset[bid] = -1;
      if (chunk == chunksm1) {
        compressed[0] = (((long)(st + loc)) << 32) + (0x43504d00 - 1) + dim;
      }
      top = loc;
      start = st;
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (tid < top) {
      compressed[start + tid] = sbuf2[tid];
    }
  }
}

/*****************************************************************************
This is the GPU decompression kernel, which requires 1024 threads per block
and should be launched with as many blocks as the GPU can run simultaneously.

Inputs
------
compressed: the input array holding the compressed data in integer format
goffset: a temporary array with m elements where m = number of thread blocks

The upper half of the first element must hold the value n, i.e., the number
of doubles that the data will generate upon decompression.

Output
------
decompressed: the output array holding the decompressed data in integer format

The output array needs to provide space for n elements has to be cast to an
array of doubles before it can be used.
*****************************************************************************/

template <int WarpSize>
static 
void MPCdecompress(
  long* __restrict const compressed, 
  long* __restrict const decompressed,
  volatile int* __restrict const goffset,
  sycl::nd_item<1> &item,
  int &start,
  int &top,
  long *__restrict sbuf1,
  long *__restrict sbuf2)
{
  const int dim = (compressed[0] & 31) + 1;
  const int n = compressed[0] >> 32;
  const int tid = item.get_local_id(0);
  const int lanex = tid & 63;
  const int warpx = tid & 0x3c0;
  const int bid = item.get_group(0);
  const int gdim = item.get_group_range(0);
  const int bid1 = ((bid + 1) == gdim) ? 0 : (bid + 1);
  const int init = 1 + (n + 63) / 64;
  const int nru = (n - 1) | 63;
  const int chunksm1 = ((n + (TPB - 1)) / TPB) - 1;

  for (int chunk = bid; chunk <= chunksm1; chunk += gdim) {
    const int idx = tid + chunk * TPB;

    int flag = 0;
    if (idx <= nru) {
      flag = (compressed[1 + idx / 64] >> lanex) & 1;
    }
    int loc = flag;

    item.barrier(sycl::access::fence_space::local_space);

    prefixsum<WarpSize>(loc, (int *)sbuf1, item);

    if (tid == (TPB - 1)) {
      int st = init;
      if (chunk > 0) {
        do {
          st = goffset[bid];
        } while (st < 0);  // busy waiting
      }
      goffset[bid1] = st + loc;
      goffset[bid] = -1;
      top = loc;
      start = st;
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (tid < top) {
      sbuf2[tid] = compressed[start + tid];
    }

    item.barrier(sycl::access::fence_space::local_space);

    long v2 = 0;
    if (flag != 0) {
      v2 = sbuf2[loc - 1];
    }

    prefixsumlong<WarpSize>(v2, sbuf1, item);

    sbuf2[tid] = v2;

    item.barrier(sycl::access::fence_space::local_space);

    long v1 = 0;
    for (int i = 63; i >= 0; i--) {
      v1 = (v1 << 1) + ((sbuf2[warpx + i] >> lanex) & 1);
    }

    prefixsumdimlong<WarpSize>(v1, sbuf1, dim, item);

    if (idx < n) {
      decompressed[idx] = v1;
    }
  }
}

int main(int argc, char *argv[])
{
  printf("MPC - Massively Parallel Compression [double] (%s)\n", __FILE__);
  printf("Copyright 2015-2020 Texas State University\n\n");

  assert(sizeof(long) == sizeof(double));

  if ((argc != 2) && (argc != 3)) {
    printf("usage: %s file [dimension]\n", argv[0]);
    exit(-1);
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int wgs = q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (wgs < TPB) {
    printf("The max work group size supported is less than %d required for the program\n", TPB);
    exit(-1);
  }
  const int SMs = q.get_device().get_info<sycl::info::device::max_compute_units>();
  const int blocks = SMs * 2;

  int dim, insize, outsize;

  string name = argv[1];
  long* const input = readFile(name.c_str(), insize);

  if (argc == 3) {
    dim = atoi(argv[2]);
    // Worst-case output size the compressor can emit: a header word, one bitmap
    // word per 64 inputs, and up to one residual word per thread. Each of the
    // ceil(insize/TPB) chunks launches TPB threads, so the residual region can
    // grow to chunks*TPB words. Small inputs still transpose full 64-word
    // bit-planes from the padded tail, so sizing residuals by insize alone
    // under-allocates and lets MPCcompress write past d_out.
    const int chunks = (insize + TPB - 1) / TPB;
    outsize = 1 + (insize + 63) / 64 + chunks * TPB;
  } else {
    assert(((input[0] >> 8) & 0xffffff) == 0x43504d);
    dim = (input[0] & 31) + 1;
    outsize = input[0] >> 32;
  }
  assert(0 < dim);  assert(dim <= 32);
  long* const output = new long[outsize];

  long* d_in, * d_out;
  int* d_offs;
  d_in = sycl::malloc_device<long>(insize, q);
  d_out = sycl::malloc_device<long>(outsize, q);
  d_offs = sycl::malloc_device<int>(blocks, q);
  q.memcpy(d_in, input, insize * sizeof(long)).wait();

  sycl::range<1> gws (blocks * TPB);
  sycl::range<1> lws (TPB);

  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int WarpSize = *r;

  if (argc == 3) {
    auto start = std::chrono::steady_clock::now();
    q.memset(d_offs, -1, blocks * sizeof(int)).wait();

    if (WarpSize == 64) { 
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 0> sm_start(cgh);
        sycl::local_accessor<int, 0> sm_top(cgh);
        sycl::local_accessor<long, 1> sbuf1(TPB, cgh);
        sycl::local_accessor<long, 1> sbuf2(TPB, cgh);

        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          MPCcompress<64>(insize, d_in, d_out, d_offs, dim, item,
                          sm_start, sm_top,
                          sbuf1.get_multi_ptr<sycl::access::decorated::no>().get(),
                          sbuf2.get_multi_ptr<sycl::access::decorated::no>().get());
        });
      }).wait();
    } else {
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 0> sm_start(cgh);
        sycl::local_accessor<int, 0> sm_top(cgh);
        sycl::local_accessor<long, 1> sbuf1(TPB, cgh);
        sycl::local_accessor<long, 1> sbuf2(TPB, cgh);

        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          MPCcompress<32>(insize, d_in, d_out, d_offs, dim, item,
                          sm_start, sm_top,
                          sbuf1.get_multi_ptr<sycl::access::decorated::no>().get(),
                          sbuf2.get_multi_ptr<sycl::access::decorated::no>().get());
        });
      }).wait();
    }

    auto end = std::chrono::steady_clock::now();

    q.memcpy(output, d_out, sizeof(long)).wait();
    outsize = output[0] >> 32;

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("compression time: %.2f ms\n", time * 1e-6);
    printf("compression throughput: %.3f GB/s\n", 1.0 * sizeof(long) * insize / time);
    printf("compression ratio: %.3f\n\n", 1.0 * insize / outsize);

    q.memcpy(output, d_out, outsize * sizeof(long)).wait();
    output[0] = (((long)insize) << 32) + (0x43504d00 - 1) + dim;

    name = "compression.txt";
    
  } else {

    auto start = std::chrono::steady_clock::now();
    q.memset(d_offs, -1, blocks * sizeof(int)).wait();

    if (WarpSize == 64) { 
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 0> sm_start(cgh);
        sycl::local_accessor<int, 0> sm_top(cgh);
        sycl::local_accessor<long, 1> sbuf1(TPB, cgh);
        sycl::local_accessor<long, 1> sbuf2(TPB, cgh);

        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          MPCdecompress<64>(
              d_in, d_out, d_offs, item,
              sm_start, sm_top,
              sbuf1.get_multi_ptr<sycl::access::decorated::no>().get(),
              sbuf2.get_multi_ptr<sycl::access::decorated::no>().get());
        });
      }).wait();
    } else {
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 0> sm_start(cgh);
        sycl::local_accessor<int, 0> sm_top(cgh);
        sycl::local_accessor<long, 1> sbuf1(TPB, cgh);
        sycl::local_accessor<long, 1> sbuf2(TPB, cgh);

        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          MPCdecompress<32>(
              d_in, d_out, d_offs, item,
              sm_start, sm_top,
              sbuf1.get_multi_ptr<sycl::access::decorated::no>().get(),
              sbuf2.get_multi_ptr<sycl::access::decorated::no>().get());
        });
      }).wait();
    }

    auto end = std::chrono::steady_clock::now();

    q.memcpy(output, d_out, outsize * sizeof(long)).wait();

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("decompression time: %.2f ms\n", time * 1e-6);
    printf("decompression throughput: %.3f GB/s\n\n", 1.0 * sizeof(long) * outsize / time);

    name = "decompression.txt";
  }

  writeFile(name.c_str(), output, outsize);

  delete [] output;
  delete [] input;

  sycl::free(d_offs, q);
  sycl::free(d_out, q);
  sycl::free(d_in, q);

  return 0;
}
