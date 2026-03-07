/*
ECL-CC code: ECL-CC is a connected components graph algorithm. The CUDA
implementation thereof is quite fast. It operates on graphs stored in
binary CSR format.

Copyright (c) 2017-2020, Texas State University. All rights reserved.

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

Authors: Jayadharini Jaiganesh and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/.

Publication: This work is described in detail in the following paper.
Jayadharini Jaiganesh and Martin Burtscher. A High-Performance Connected
Components Implementation for GPUs. Proceedings of the 2018 ACM International
Symposium on High-Performance Parallel and Distributed Computing, pp. 92-104.
June 2018.
*/

#include <stdlib.h>
#include <stdio.h>
#include <set>
#include <chrono>
#include <sycl/sycl.hpp>
#include "graph.h"

static const int ThreadsPerBlock = 256;

inline int atomicCAS(int &val, int expected, int desired) 
{
  int expected_value = expected;
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(val);
  atm.compare_exchange_strong(expected_value, desired);
  return expected_value;
}

inline int atomicAdd(int *val, int operand) 
{
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(*val);
  return atm.fetch_add(operand);
}

/* initialize with first smaller neighbor ID */

static 
void init(sycl::nd_item<1> &item,
          const int nodes,
          const int* const __restrict nidx,
          const int* const __restrict nlist,
                int* const __restrict nstat,
                int *topL,
                int *posL,
                int *topH,
                int *posH)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    const int beg = nidx[v];
    const int end = nidx[v + 1];
    int m = v;
    int i = beg;
    while ((m == v) && (i < end)) {
      m = sycl::min(m, nlist[i]);
      i++;
    }
    nstat[v] = m;
  }

  if (from == 0) { *topL = 0; *posL = 0; *topH = nodes - 1; *posH = nodes - 1; }
}

/* intermediate pointer jumping */

static inline int representative(const int idx, int* const __restrict nstat)
{
  int curr = nstat[idx];
  if (curr != idx) {
    int next, prev = idx;
    while (curr > (next = nstat[curr])) {
      nstat[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

/* process low-degree vertices at thread granularity and fill worklists */

static 
void compute1(sycl::nd_item<1> &item,
              const int nodes,
              const int* const __restrict nidx,
              const int* const __restrict nlist,
                    int* const __restrict nstat,
                    int* const __restrict wl,
                    int *const __restrict topL,
                    int *const __restrict topH)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    const int vstat = nstat[v];
    if (v != vstat) {
      const int beg = nidx[v];
      const int end = nidx[v + 1];
      int deg = end - beg;
      if (deg > 16) {
        int idx;
        if (deg <= 352) {
          idx = atomicAdd(topL, 1);
        } else {
          idx = atomicAdd(topH, -1);
        }
        wl[idx] = v;
      } else {
        int vstat = representative(v, nstat);
        for (int i = beg; i < end; i++) {
          const int nli = nlist[i];
          if (v > nli) {
            int ostat = representative(nli, nstat);
            bool repeat;
            do {
              repeat = false;
              if (vstat != ostat) {
                int ret;
                if (vstat < ostat) {
                  if ((ret = atomicCAS(nstat[ostat], ostat, vstat)) != ostat) {
                    ostat = ret;
                    repeat = true;
                  }
                } else {
                  if ((ret = atomicCAS(nstat[vstat], vstat, ostat)) != vstat) {
                    vstat = ret;
                    repeat = true;
                  }
                }
              }
            } while (repeat);
          }
        }
      }
    }
  }
}

/* process medium-degree vertices at warp granularity */

static 
void compute2(sycl::nd_item<1> &item,
              const int nodes,
              const int* const __restrict nidx,
              const int* const __restrict nlist,
                    int* const __restrict nstat,
              const int* const __restrict wl,
              const int *const __restrict topL,
                    int *const __restrict posL)
{
  auto sg = item.get_sub_group();
  const int lane = sg.get_local_linear_id();

  int idx;
  if (lane == 0) idx = atomicAdd(posL, 1);
  idx = sycl::select_from_group(sg, idx, 0);
  while (idx < *topL) {
    const int v = wl[idx];
    int vstat = representative(v, nstat);
    for (int i = nidx[v] + lane; i < nidx[v + 1]; i += sg.get_max_local_range()[0]) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(nstat[ostat], ostat, vstat)) != ostat) {
                ostat = ret;
                repeat = true;
              }
            } else {
              if ((ret = atomicCAS(nstat[vstat], vstat, ostat)) != vstat) {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
    if (lane == 0) idx = atomicAdd(posL, 1);
    idx = sycl::select_from_group(sg, idx, 0);
  }
}

/* process high-degree vertices at block granularity */

static 
void compute3(sycl::nd_item<1> &item,
              const int nodes,
              const int* const __restrict nidx,
              const int* const __restrict nlist,
                    int* const __restrict nstat,
              const int* const __restrict wl,
              const int* const __restrict topH,
                    int* const __restrict posH,
                    int& vB)
{

  if (item.get_local_id(0) == 0) {
    const int idx = atomicAdd(posH, -1);
    vB = (idx > *topH) ? wl[idx] : -1;
  }
  item.barrier(sycl::access::fence_space::local_space);

  while (vB >= 0) {
    const int v = (vB);
    item.barrier(sycl::access::fence_space::local_space);

    int vstat = representative(v, nstat);
    for (int i = nidx[v] + item.get_local_id(0); i < nidx[v + 1];
         i += ThreadsPerBlock) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(nstat[ostat], ostat, vstat)) != ostat) {
                ostat = ret;
                repeat = true;
              }
            } else {
              if ((ret = atomicCAS(nstat[vstat], vstat, ostat)) != vstat) {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
    if (item.get_local_id(0) == 0) {
      const int idx = atomicAdd(posH, -1);
      vB = (idx > *topH) ? wl[idx] : -1;
    }

    item.barrier(sycl::access::fence_space::local_space);
  }
}

/* link all vertices to sink */

static 
void flatten(sycl::nd_item<1> &item,
             const int nodes,
             const int* const __restrict nidx,
             const int* const __restrict nlist,
                   int* const __restrict nstat)
{
  const int from = item.get_global_id(0);
  const int incr = item.get_group_range(0) * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    int next, vstat = nstat[v];
    const int old = vstat;
    while (vstat > (next = nstat[vstat])) {
      vstat = next;
    }
    if (old != vstat) nstat[v] = vstat;
  }
}

static void computeCC(const int repeat,
                      const int nodes, const int edges,
                      const int *const __restrict nidx,
                      const int *const __restrict nlist,
                            int *const __restrict nstat) 
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int SMs = q.get_device().get_info<sycl::info::device::max_compute_units>();
  // not supported by SYCL yet
  const int mTSM = 2048;
  const int blocks = SMs * mTSM / ThreadsPerBlock;
  printf("Number of multiprocessor = %d\n", SMs);
  printf("Max threads per multiprocessor = %d\n", mTSM);
  printf("Number of thread blocks in a grid = %d\n", blocks);

  int *topL_d = sycl::malloc_device<int>(1, q);
  int *posL_d = sycl::malloc_device<int>(1, q);
  int *topH_d = sycl::malloc_device<int>(1, q);
  int *posH_d = sycl::malloc_device<int>(1, q);

  int *nidx_d = sycl::malloc_device<int>(nodes+1, q);
  q.memcpy(nidx_d, nidx, (nodes + 1) * sizeof(int));

  int *nlist_d = sycl::malloc_device<int>(edges, q);
  q.memcpy(nlist_d, nlist, edges * sizeof(int));

  int *nstat_d = sycl::malloc_device<int>(nodes, q);

  int *wl_d = sycl::malloc_device<int>(nodes, q);

  sycl::range<1> gws (blocks * ThreadsPerBlock);
  sycl::range<1> lws (ThreadsPerBlock);

  q.wait();

  auto start = std::chrono::high_resolution_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class initialize>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        init(item, nodes, nidx_d, nlist_d, nstat_d, 
             topL_d, posL_d, topH_d, posH_d);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class compute_low>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        compute1(item, nodes, nidx_d, nlist_d, 
                 nstat_d, wl_d, topL_d, topH_d);
      });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class compute_med>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item)
      {
        compute2(item, nodes, nidx_d, nlist_d,
                 nstat_d, wl_d, topL_d, posL_d);
      });
    });

    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int, 0> vB (cgh);
      cgh.parallel_for<class compute_high>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        compute3(item, nodes, nidx_d, nlist_d,
                 nstat_d, wl_d, topH_d, posH_d, vB);
      });
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class link>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        flatten(item, nodes, nidx_d, nlist_d, nstat_d);
      });
    });
  }

  q.wait();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  float runtime = elapsed_seconds.count() / repeat;

  printf("compute time: %.4f s\n", runtime);
  printf("throughput: %.3f Mnodes/s\n", nodes * 0.000001 / runtime);
  printf("throughput: %.3f Medges/s\n", edges * 0.000001 / runtime);

  q.memcpy(nstat, nstat_d, nodes * sizeof(int)).wait();
  sycl::free(wl_d, q);
  sycl::free(nstat_d, q);
  sycl::free(nlist_d, q);
  sycl::free(nidx_d, q);
  sycl::free(topL_d, q);
  sycl::free(posL_d, q);
  sycl::free(topH_d, q);
  sycl::free(posH_d, q);
}

static void verify(const int v, const int id, const int* const __restrict nidx, 
                   const int* const __restrict nlist, int* const __restrict nstat,
                   bool &ok)
{
  if (nstat[v] >= 0) {
    if (nstat[v] != id) {
      fprintf(stderr, "ERROR: found incorrect ID value\n\n");
      ok = false;
      return;
    }
    nstat[v] = -1;
    for (int i = nidx[v]; i < nidx[v + 1]; i++) {
      verify(nlist[i], id, nidx, nlist, nstat, ok);
    }
  }
}

int main(int argc, char* argv[])
{
  printf("ECL-CC v1.1 (%s)\n", __FILE__);
  printf("Copyright 2017-2020 Texas State University\n");

  if (argc != 3) {
    fprintf(stderr, "USAGE: %s <input_file_name> <repeat>\n\n", argv[0]);
    exit(-1);
  }

  ECLgraph g = readECLgraph(argv[1]);
  const int repeat = atoi(argv[2]);

  int* nodestatus = (int*) malloc (sizeof(int) * g.nodes);

  if (nodestatus == NULL) {
    fprintf(stderr, "ERROR: nodestatus - host memory allocation failed\n\n");
    exit(-1);
  }

  printf("input graph: %d nodes and %d edges (%s)\n", g.nodes, g.edges, argv[1]);
  printf("average degree: %.2f edges per node\n", 1.0 * g.edges / g.nodes);
  int mindeg = g.nodes;
  int maxdeg = 0;
  for (int v = 0; v < g.nodes; v++) {
    int deg = g.nindex[v + 1] - g.nindex[v];
    mindeg = std::min(mindeg, deg);
    maxdeg = std::max(maxdeg, deg);
  }
  printf("minimum degree: %d edges\n", mindeg);
  printf("maximum degree: %d edges\n", maxdeg);

  computeCC(repeat, g.nodes, g.edges, g.nindex, g.nlist, nodestatus);

  std::set<int> s1;
  for (int v = 0; v < g.nodes; v++) {
    s1.insert(nodestatus[v]);
  }
  printf("number of connected components: %zu\n", s1.size());

  /* verification code (may need extra runtime stack space due to deep recursion) */
  bool ok = true;
  for (int v = 0; v < g.nodes; v++) {
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      if (nodestatus[g.nlist[i]] != nodestatus[v]) {
        fprintf(stderr, "ERROR: found adjacent nodes in different components\n\n");
        ok = false; break;
      }
    }
  }

  for (int v = 0; v < g.nodes; v++) {
    if (nodestatus[v] < 0) {
      fprintf(stderr, "ERROR: found negative component number\n\n");
      ok = false; break;
    }
  }

  std::set<int> s2;
  size_t count = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (nodestatus[v] >= 0) {
      count++;
      s2.insert(nodestatus[v]);
      verify(v, nodestatus[v], g.nindex, g.nlist, nodestatus, ok);
    }
  }
  if (s1.size() != s2.size()) {
    fprintf(stderr, "ERROR: number of components do not match\n\n");  
    ok = false;
  }

  if (s1.size() != count) {
    fprintf(stderr, "ERROR: component IDs are not unique\n\n");
    ok = false;
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
  free(nodestatus);
  freeECLgraph(g);

  return 0;
}
