/*
   ECL-MIS code: ECL-MIS is a maximal independent set algorithm that is quite
   fast and produces relatively large sets. It operates on graphs stored in
   binary CSR format.

   Copyright (c) 2017-2020, Texas State University. All rights reserved. Patented.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted (subject to the limitations in the disclaimer
   below) provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY
 BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.

Authors: Martin Burtscher and Sindhu Devale

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-MIS/.

Publication: This work is described in detail in the following paper.
Martin Burtscher, Sindhu Devale, Sahar Azimi, Jayadharini Jaiganesh, and
Evan Powers. A High-Quality and Fast Maximal Independent Set Implementation
for GPUs. ACM Transactions on Parallel Computing, Vol. 5, No. 2, Article 8
(27 pages). December 2018.
 */


#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include "graph.h"

static const int ThreadsPerBlock = 256;

typedef unsigned char stattype;
static const stattype in = 0xfe;
static const stattype out = 0;

/* main computation kernel */

__global__ 
void findmins(const int nodes, 
    const int* const __restrict nidx,
    const int* const __restrict nlist,
    volatile stattype* const __restrict nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  int missing;
  do {
    missing = 0;
    for (int v = from; v < nodes; v += incr) {
      const stattype nv = nstat[v];
      if (nv & 1) {
        int i = nidx[v];
        while ((i < nidx[v + 1]) && ((nv > nstat[nlist[i]]) || ((nv == nstat[nlist[i]]) && (v > nlist[i])))) {
          i++;
        }
        if (i < nidx[v + 1]) {
          missing = 1;
        } else {
          for (int i = nidx[v]; i < nidx[v + 1]; i++) {
            nstat[nlist[i]] = out;
          }
          nstat[v] = in;
        }
      }
    }
  } while (missing != 0);
}

/* hash function to generate random values */

// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
__device__ 
unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

/* prioritized-selection initialization kernel */

__global__
void init(const int nodes, 
    const int edges, 
    const int* const __restrict nidx,
    stattype* const __restrict nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  const float avg = (float)edges / nodes;
  const float scaledavg = ((in / 2) - 1) * avg;

  for (int i = from; i < nodes; i += incr) {
    stattype val = in;
    const int degree = nidx[i + 1] - nidx[i];
    if (degree > 0) {
      float x = degree - (hash(i) * 0.00000000023283064365386962890625f);
      int res = int(scaledavg / (avg + x));
      val = (res + res) | 1;
    }
    nstat[i] = val;
  }
}

void computeMIS(
    const int repeat,
    const int nodes,
    const int edges,
    const int* const __restrict nidx,
    const int* const __restrict nlist,
    stattype* const __restrict nstat)
{
  int* nidx_d;
  int* nlist_d;
  stattype* nstat_d;

  if (hipSuccess != hipMalloc((void **)&nidx_d, (nodes + 1) * sizeof(int))) {
    fprintf(stderr, "ERROR: could not allocate nidx_d\n\n");
  }
  if (hipSuccess != hipMalloc((void **)&nlist_d, edges * sizeof(int))) {
    fprintf(stderr, "ERROR: could not allocate nlist_d\n\n");
  }
  if (hipSuccess != hipMalloc((void **)&nstat_d, nodes * sizeof(stattype))) {
    fprintf(stderr, "ERROR: could not allocate nstat_d\n\n");
  }
  if (hipSuccess != hipMemcpy(nidx_d, nidx, (nodes + 1) * sizeof(int), hipMemcpyHostToDevice)) {
    fprintf(stderr, "ERROR: copying to device failed\n\n");
  }
  if (hipSuccess != hipMemcpy(nlist_d, nlist, edges * sizeof(int), hipMemcpyHostToDevice)) {
    fprintf(stderr, "ERROR: copying to device failed\n\n");
  }

  const int blocks = 24;

  hipDeviceSynchronize();

  auto start = std::chrono::high_resolution_clock::now();

  for (int n = 0; n < repeat; n++) {
    init<<<blocks, ThreadsPerBlock>>>(nodes, edges, nidx_d, nstat_d);

    findmins<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d);
  }

  hipDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  float runtime = (float)elapsed_seconds.count() / repeat;
  printf("compute time: %.6f s\n", runtime);
  printf("throughput: %.6f Mnodes/s\n", nodes * 0.000001 / runtime);
  printf("throughput: %.6f Medges/s\n", edges * 0.000001 / runtime);

  if (hipSuccess != hipMemcpy(nstat, nstat_d, nodes * sizeof(stattype), hipMemcpyDeviceToHost)) {
    fprintf(stderr, "ERROR: copying from device failed\n\n");
  }

  hipFree(nstat_d);
  hipFree(nlist_d);
  hipFree(nidx_d);
}

int main(int argc, char* argv[])
{
  printf("ECL-MIS v1.3 (%s)\n", __FILE__);
  printf("Copyright 2017-2020 Texas State University\n");

  if (argc != 3) {
    fprintf(stderr, "USAGE: %s <input_file_name> <repeat>\n\n", argv[0]);
    exit(-1);
  }

  ECLgraph g = readECLgraph(argv[1]);
  verifyUndirectedECLgraph(g);
  printf("configuration: %d nodes and %d edges (%s)\n", g.nodes, g.edges, argv[1]);
  printf("average degree: %.2f edges per node\n", 1.0 * g.edges / g.nodes);

  stattype* nstatus = (stattype*)malloc(g.nodes * sizeof(nstatus[0]));

  if (nstatus == NULL) {
    fprintf(stderr, "ERROR: could not allocate nstatus\n\n");
  }
  else {
    const int repeat = atoi(argv[2]);

    computeMIS(repeat, g.nodes, g.edges, g.nindex, g.nlist, nstatus);

    /* result verification code */

    for (int v = 0; v < g.nodes; v++) {
      if ((nstatus[v] != in) && (nstatus[v] != out)) {
        fprintf(stderr, "ERROR: found unprocessed node in graph\n\n");
        break;
      }
      if (nstatus[v] == in) {
        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
          if (nstatus[g.nlist[i]] == in) {
            fprintf(stderr, "ERROR: found adjacent nodes in MIS\n\n");
            break;
          }
        }
      } else {
        int flag = 0;
        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
          if (nstatus[g.nlist[i]] == in) {
            flag = 1;
          }
        }
        if (flag == 0) {
          fprintf(stderr, "ERROR: set is not maximal\n\n");
          break;
        }
      }
    }
  }

  freeECLgraph(g);
  if (nstatus != NULL) free(nstatus);
  return 0;
}
