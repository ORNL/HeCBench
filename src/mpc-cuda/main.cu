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
#include "utils.h"

#define TPB 1024  /* do not change */

#if (CUDART_VERSION >= 9000)
#define __shfl_up(v, d) __shfl_up_sync(0xffffffff, v, d)
#endif

static inline __device__
void prefixsum(int &val, int sbuf[TPB])
{
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  for (int d = 1; d < 32; d *= 2) {
    int tmp = __shfl_up(val, d);
    if (lane >= d) val += tmp;
  }
  if (lane == 31) sbuf[warp] = val;

  __syncthreads();
  if (warp == 0) {
    int v = sbuf[lane];
    for (int d = 1; d < 32; d *= 2) {
      int tmp = __shfl_up(v, d);
      if (lane >= d) v += tmp;
    }
    sbuf[lane] = v;
  }

  __syncthreads();
  if (warp > 0) {
    val += sbuf[warp - 1];
  }
}

static inline __device__
void prefixsumlong(long &val, long sbuf[TPB])
{
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  for (int d = 1; d < 32; d *= 2) {
    unsigned int tmpl = __shfl_up((int)val, d);
    long tmph = __shfl_up((int)(val >> 32), d);
    if (lane >= d) val += (tmph << 32) + tmpl;
  }
  if (lane == 31) sbuf[warp] = val;

  __syncthreads();
  if (warp == 0) {
    long v = sbuf[lane];
    for (int d = 1; d < 32; d *= 2) {
      unsigned int tmpl = __shfl_up((int)v, d);
      long tmph = __shfl_up((int)(v >> 32), d);
      if (lane >= d) v += (tmph << 32) + tmpl;
    }
    sbuf[lane] = v;
  }

  __syncthreads();
  if (warp > 0) {
    val += sbuf[warp - 1];
  }
}

static inline __device__
void prefixsumdimlong(long &val, long sbuf[TPB], const unsigned char dim)
{
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int tix = (warp * dim) + (tid % dim);

  for (int d = dim; d < 32; d *= 2) {
    unsigned int tmpl = __shfl_up((int)val, d);
    long tmph = __shfl_up((int)(val >> 32), d);
    if (lane >= d) val += (tmph << 32) + tmpl;
  }
  if ((lane + dim) > 31) sbuf[tix] = val;

  __syncthreads();
  if (warp < dim) {
    const int idx = (lane * dim) + warp;
    long v = sbuf[idx];
    for (int d = 1; d < 32; d *= 2) {
      unsigned int tmpl = __shfl_up((int)v, d);
      long tmph = __shfl_up((int)(v >> 32), d);
      if (lane >= d) v += (tmph << 32) + tmpl;
    }
    sbuf[idx] = v;
  }

  __syncthreads();
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

static __global__ __launch_bounds__(1024, 2)
void MPCcompress(
  const int n,
  long* __restrict__ const original,
  long* __restrict__ const compressed,
  volatile int* __restrict__ const goffset,
  unsigned char dim)
{
  const int tid = threadIdx.x;
  const int tidm1 = tid - 1;
  const int tidmdim = tid - dim;
  const int lanex = tid & 63;
  const int warpx = tid & 0x3c0;
  const int bid = blockIdx.x;
  const int gdim = gridDim.x;
  const int bid1 = ((bid + 1) == gdim) ? 0 : (bid + 1);
  const int init = 1 + (n + 63) / 64;
  const int chunksm1 = ((n + (TPB - 1)) / TPB) - 1;

  __shared__ int start, top;
  __shared__ long sbuf1[TPB], sbuf2[TPB];

  for (int chunk = bid; chunk <= chunksm1; chunk += gdim) {
    const int idx = tid + chunk * TPB;

    long v1 = 0;
    if (idx < n) {
      v1 = original[idx];
      sbuf1[tid] = v1;
    }

    __syncthreads();
    if (tid >= dim) {
      if (idx < n) {
        v1 -= sbuf1[tidmdim];
      }
    }
    sbuf2[tid] = v1;

    __syncthreads();
    long v2 = 0;

    for (int i = 63; i >= 0; i--) {
      v2 = (v2 << 1) + ((sbuf2[warpx + i] >> lanex) & 1);
    }
    sbuf1[tid] = v2;

    __syncthreads();
    if (tid > 0) {
      v2 -= sbuf1[tidm1];
    }

    int loc = 0;
    if (v2 != 0) loc = 1;

#if (CUDART_VERSION < 9000)
    unsigned int bitmap = __ballot(loc);
#else
    unsigned int bitmap = __ballot_sync(0xffffffff, loc);
#endif

    if (lanex == 32) {
      sbuf2[tid] = bitmap;
    }

    __syncthreads();
    if (lanex == 0) {
      if (idx < n) compressed[1 + idx / 64] = (sbuf2[tid + 32] << 32) + bitmap;
    }

    prefixsum(loc, (int*)sbuf1);

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

    __syncthreads();
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

static __global__ __launch_bounds__(1024, 2)
void MPCdecompress(
  long* __restrict__ const compressed,
  long* __restrict__ const decompressed,
  volatile int* __restrict__ const goffset)
{
  const int dim = (compressed[0] & 31) + 1;
  const int n = compressed[0] >> 32;
  const int tid = threadIdx.x;
  const int lanex = tid & 63;
  const int warpx = tid & 0x3c0;
  const int bid = blockIdx.x;
  const int gdim = gridDim.x;
  const int bid1 = ((bid + 1) == gdim) ? 0 : (bid + 1);
  const int init = 1 + (n + 63) / 64;
  const int nru = (n - 1) | 63;
  const int chunksm1 = ((n + (TPB - 1)) / TPB) - 1;

  __shared__ int start, top;
  __shared__ long sbuf1[TPB], sbuf2[TPB];

  for (int chunk = bid; chunk <= chunksm1; chunk += gdim) {
    const int idx = tid + chunk * TPB;

    int flag = 0;
    if (idx <= nru) {
      flag = (compressed[1 + idx / 64] >> lanex) & 1;
    }
    int loc = flag;

    __syncthreads();
    prefixsum(loc, (int*)sbuf1);

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

    __syncthreads();
    if (tid < top) {
      sbuf2[tid] = compressed[start + tid];
    }

    __syncthreads();
    long v2 = 0;
    if (flag != 0) {
      v2 = sbuf2[loc - 1];
    }

    prefixsumlong(v2, sbuf1);

    sbuf2[tid] = v2;

    __syncthreads();
    long v1 = 0;
    for (int i = 63; i >= 0; i--) {
      v1 = (v1 << 1) + ((sbuf2[warpx + i] >> lanex) & 1);
    }

    prefixsumdimlong(v1, sbuf1, dim);

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

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  const int blocks = deviceProp.multiProcessorCount * 2;
  // deprecated API
  //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  int dim, insize, outsize;

  std::string name = argv[1];
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

  long *d_in, *d_out;
  int *d_offs;
  cudaMalloc(&d_in, insize * sizeof(long));
  cudaMalloc(&d_out, outsize * sizeof(long));
  cudaMalloc(&d_offs, blocks * sizeof(int));
  cudaMemcpy(d_in, input, insize * sizeof(long), cudaMemcpyHostToDevice);

  if (argc == 3) {
    auto start = std::chrono::steady_clock::now();
    cudaMemset(d_offs, -1, blocks * sizeof(int));
    MPCcompress<<<blocks, TPB>>>(insize, d_in, d_out, d_offs, dim);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    cudaMemcpy(output, d_out, sizeof(long), cudaMemcpyDeviceToHost);
    outsize = output[0] >> 32;

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("compression time: %.2f ms\n", time * 1e-6);
    printf("compression throughput: %.3f GB/s\n", 1.0 * sizeof(long) * insize / time);
    printf("compression ratio: %.3f\n\n", 1.0 * insize / outsize);

    cudaMemcpy(output, d_out, outsize * sizeof(long), cudaMemcpyDeviceToHost);
    output[0] = (((long)insize) << 32) + (0x43504d00 - 1) + dim;

    name = "compression.txt";

  } else {

    auto start = std::chrono::steady_clock::now();
    cudaMemset(d_offs, -1, blocks * sizeof(int));
    MPCdecompress<<<blocks, TPB>>>(d_in, d_out, d_offs);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    cudaMemcpy(output, d_out, outsize * sizeof(long), cudaMemcpyDeviceToHost);

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("decompression time: %.2f ms\n", time * 1e-6);
    printf("decompression throughput: %.3f GB/s\n\n", 1.0 * sizeof(long) * outsize / time);

    name = "decompression.txt";
  }

  writeFile(name.c_str(), output, outsize);

  delete [] output;
  delete [] input;

  cudaFree(d_offs);
  cudaFree(d_out);
  cudaFree(d_in);

  return 0;
}
