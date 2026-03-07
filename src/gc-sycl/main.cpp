/*
   ECL-GC code: ECL-GC is a graph-coloring algorithm with shortcutting. The CUDA
   implementation thereof is quite fast. It operates on graphs stored in binary
   CSR format.

   Copyright (c) 2020, Texas State University. All rights reserved.

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

Authors: Ghadeer Alabandi, Evan Powers, and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-GC/.

Publication: This work is described in detail in the following paper.
Ghadeer Alabandi, Evan Powers, and Martin Burtscher. Increasing the Parallelism
of Graph Coloring via Shortcutting. Proceedings of the 2020 ACM SIGPLAN
Symposium on Principles and Practice of Parallel Programming, pp. 262-275.
February 2020.
 */

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <sycl/sycl.hpp>
#include "graph.h"

// the maximum work-group size depends on the target device
static const int ThreadsPerBlock = 512;
static const int WS = 32;  // warp size and bits per int
static const int MSB = 1 << (WS - 1);
static const int Mask = (1 << (WS / 2)) - 1;

inline int ffs(int x) {
  return (x == 0) ? 0 : sycl::ctz(x) + 1;
}

// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}


void init(const int nodes,
    const int edges, 
    const int* const __restrict nidx,
    const int* const __restrict nlist,
    int* const __restrict nlist2,
    int* const __restrict posscol,
    int* const __restrict posscol2,
    int* const __restrict color,
    int* const __restrict wl,
    int* __restrict wlsize,
    sycl::nd_item<1> &item
#ifdef SYCL_STREAM
    , const sycl::stream &out
#endif
    )
{
  const int lane = item.get_local_id(0) % WS;
  const int thread = item.get_global_id(0);
  const int threads = item.get_group_range(0) * ThreadsPerBlock;
  auto sg = item.get_sub_group();

  int maxrange = -1;
  for (int v = thread; sycl::any_of_group(sg, v < nodes); v += threads) {
    bool cond = false;
    int beg, end, pos, degv, active;
    if (v < nodes) {
      beg = nidx[v];
      end = nidx[v + 1];
      degv = end - beg;
      cond = (degv >= WS);
      if (cond) {
        auto ao = sycl::atomic_ref<int, sycl::memory_order::relaxed, \
                                   sycl::memory_scope::device,\
                                   sycl::access::address_space::generic_space>(*wlsize);
        wl[ao.fetch_add(1)] = v;
      } else {
        active = 0;
        pos = beg;
        for (int i = beg; i < end; i++) {
          const int nei = nlist[i];
          const int degn = nidx[nei + 1] - nidx[nei];
          if ((degv < degn) || ((degv == degn) && (hash(v) < hash(nei))) || 
             ((degv == degn) && (hash(v) == hash(nei)) && (v < nei))) {
            active |= (unsigned int)MSB >> (i - beg);
            pos++;
          }
        }
      }
    }

    int bal = sycl::reduce_over_group(sg,
        cond ? (0x1 << sg.get_local_linear_id()) : 0, sycl::plus<>());
    while (bal != 0) {
      const int who = ffs(bal) - 1;
      bal &= bal - 1;
      const int wv = sycl::select_from_group(sg, v, who);
      const int wbeg = sycl::select_from_group(sg, beg, who);
      const int wend = sycl::select_from_group(sg, end, who);
      const int wdegv = wend - wbeg;
      int wpos = wbeg;
      for (int i = wbeg + lane; sycl::any_of_group(sg, i < wend); i += WS) {
        int wnei;
        bool prio = false;
        if (i < wend) {
          wnei = nlist[i];
          const int wdegn = nidx[wnei + 1] - nidx[wnei];
          prio = ((wdegv < wdegn) || ((wdegv == wdegn) && (hash(wv) < hash(wnei))) || 
                 ((wdegv == wdegn) && (hash(wv) == hash(wnei)) && (wv < wnei)));
        }
        const int b = sycl::reduce_over_group( sg, 
                  prio ? (0x1 << sg.get_local_linear_id()) : 0, sycl::plus<>());
        const int offs = sycl::popcount(b & ((1 << lane) - 1));
        if (prio) nlist2[wpos + offs] = wnei;
        wpos += sycl::popcount(b);
      }
      if (who == lane) pos = wpos;
    }

    if (v < nodes) {
      const int range = pos - beg;
      maxrange = sycl::max(maxrange, range);
      color[v] = (cond || (range == 0)) ? (range << (WS / 2)) : active;
      posscol[v] = (range >= WS) ? -1 : (MSB >> range);
    }
  }
#ifdef SYCL_STREAM
  if (maxrange >= Mask) out << "too many active neighbors\n";
#endif

  for (int i = thread; i < edges / WS + 1; i += threads) posscol2[i] = -1;
}

void runLarge(const int nodes, 
    const int* const __restrict nidx,
    const int* const __restrict nlist,
    int* const __restrict posscol,
    int* const __restrict posscol2,
    volatile int* const __restrict color,
    const int* const __restrict wl,
    const int* __restrict wlsize,
    sycl::nd_item<1> &item)
{
  const int stop = *wlsize;
  if (stop != 0) {
    const int lane = item.get_local_id(0) % WS;
    const int thread = item.get_global_id(0);
    const int threads = item.get_group_range(0) * ThreadsPerBlock;
    auto sg = item.get_sub_group();
    bool again;
    do {
      again = false;
      for (int w = thread; sycl::any_of_group(sg, w < stop); w += threads) {
        bool shortcut, done, cond = false;
        int v, data, range, beg, pcol;
        if (w < stop) {
          v = wl[w];
          data = color[v];
          range = data >> (WS / 2);
          if (range > 0) {
            beg = nidx[v];
            pcol = posscol[v];
            cond = true;
          }
        }

        int bal = sycl::reduce_over_group(sg,
            cond ? (0x1 << sg.get_local_linear_id()) : 0, sycl::plus<>());
        while (bal != 0) {
          const int who = ffs(bal) - 1;
          bal &= bal - 1;
          const int wdata = sycl::select_from_group(sg, data, who);
          const int wrange = wdata >> (WS / 2);
          const int wbeg = sycl::select_from_group(sg, beg, who);
          const int wmincol = wdata & Mask;
          const int wmaxcol = wmincol + wrange;
          const int wend = wbeg + wmaxcol;
          const int woffs = wbeg / WS;
          int wpcol = sycl::select_from_group(sg, pcol, who);

          bool wshortcut = true;
          bool wdone = true;
          for (int i = wbeg + lane; sycl::any_of_group(sg, i < wend); i += WS) {
            int nei, neidata, neirange;
            if (i < wend) {
              nei = nlist[i];
              neidata = color[nei];
              neirange = neidata >> (WS / 2);
              const bool neidone = (neirange == 0);
              wdone &= neidone; //consolidated below
              if (neidone) {
                const int neicol = neidata;
                if (neicol < WS) {
                  wpcol &= ~((unsigned int)MSB >> neicol); //consolidated below
                } else {
                  if ((wmincol <= neicol) && (neicol < wmaxcol) && ((posscol2[woffs + neicol / WS] << (neicol % WS)) < 0)) {
                     auto ao = sycl::atomic_ref<int, sycl::memory_order::relaxed, \
                                                sycl::memory_scope::device,\
                                                sycl::access::address_space::generic_space>(posscol2[woffs + neicol / WS]);
                     ao.fetch_and(~((unsigned int)MSB >> (neicol % WS)));
                  }
                }
              } else {
                const int neimincol = neidata & Mask;
                const int neimaxcol = neimincol + neirange;
                if ((neimincol <= wmincol) && (neimaxcol >= wmincol)) wshortcut = false; //consolidated below
              }
            }
          }
          wshortcut = sycl::all_of_group(sg, wshortcut);
          wdone = sycl::all_of_group(sg, wdone);
          wpcol &= sycl::permute_group_by_xor(sg, wpcol, 1);
          wpcol &= sycl::permute_group_by_xor(sg, wpcol, 2);
          wpcol &= sycl::permute_group_by_xor(sg, wpcol, 4);
          wpcol &= sycl::permute_group_by_xor(sg, wpcol, 8);
          wpcol &= sycl::permute_group_by_xor(sg, wpcol, 16);
          if (who == lane) pcol = wpcol;
          if (who == lane) done = wdone;
          if (who == lane) shortcut = wshortcut;
        }

        if (w < stop) {
          if (range > 0) {
            const int mincol = data & Mask;
            int val = pcol, mc = 0;
            if (pcol == 0) {
              const int offs = beg / WS;
              mc = sycl::max(1, (int)(mincol / WS));
              while ((val = posscol2[offs + mc]) == 0) mc++;
            }
            int newmincol = mc * WS + sycl::clz(val);
            if (mincol != newmincol) shortcut = false;
            if (shortcut || done) {
              pcol = (newmincol < WS) ? ((unsigned int)MSB >> newmincol) : 0;
            } else {
              const int maxcol = mincol + range;
              const int range = maxcol - newmincol;
              newmincol = (range << (WS / 2)) | newmincol;
              again = true;
            }
            posscol[v] = pcol;
            color[v] = newmincol;
          }
        }
      }
    } while (sycl::any_of_group(sg, again));
  }
}

  
void runSmall(const int nodes,
    const int* const __restrict nidx,
    const int* const __restrict nlist,
    volatile int* const __restrict posscol,
    int* const __restrict color,
    sycl::nd_item<1> &item)
{
  const int thread = item.get_global_id(0);
  const int threads = item.get_group_range(0) * ThreadsPerBlock;

  bool again;
  do {
    again = false;
    for (int v = thread; v < nodes; v += threads) {
      int pcol = posscol[v];
      if (sycl::popcount(pcol) > 1) {
        const int beg = nidx[v];
        int active = color[v];
        int allnei = 0;
        int keep = active;
        do {
          const int old = active;
          active &= active - 1;
          const int curr = old ^ active;
          const int i = beg + sycl::clz((int)curr);
          const int nei = nlist[i];
          const int neipcol = posscol[nei];
          allnei |= neipcol;
          if ((pcol & neipcol) == 0) {
            pcol &= pcol - 1;
            keep ^= curr;
          } else if (sycl::popcount(neipcol) == 1) {
            pcol ^= neipcol;
            keep ^= curr;
          }
        } while (active != 0);
        if (keep != 0) {
          const int best = (unsigned int)MSB >> sycl::clz(pcol);
          if ((best & ~allnei) != 0) {
            pcol = best;
            keep = 0;
          }
        }
        again |= keep;
        if (keep == 0) keep = sycl::clz(pcol);
        color[v] = keep;
        posscol[v] = pcol;
      }
    }
  } while (again);
}

int main(int argc, char *argv[]) {
  printf("ECL-GC v1.2 (%s)\n", __FILE__);
  printf("Copyright 2020 Texas State University\n\n");

  if (argc != 3) {printf("USAGE: %s <input_file_name> <repeat>\n\n", argv[0]);  exit(-1);}
  if (WS != 32) {printf("ERROR: warp size must be 32\n\n");  exit(-1);}
  if (WS != sizeof(int) * 8) {printf("ERROR: bits per word must match warp size\n\n");  exit(-1);}
  if ((ThreadsPerBlock < WS) || ((ThreadsPerBlock % WS) != 0)) {
    printf("ERROR: threads per block must be a multiple of the warp size\n\n");
    exit(-1);
  }
  if ((ThreadsPerBlock & (ThreadsPerBlock - 1)) != 0) {
    printf("ERROR: threads per block must be a power of two\n\n");
    exit(-1);
  }

  printf("input: %s\n", argv[1]);
  ECLgraph g = readECLgraph(argv[1]);

  const int nodes = g.nodes;
  const int edges = g.edges;

  printf("nodes: %d\n", nodes);
  printf("edges: %d\n", edges);
  printf("avg degree: %.2f\n", 1.0 * edges / nodes);

  const int repeat = atoi(argv[2]);

  int* const color = new int [nodes];

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int *nidx_d = sycl::malloc_device<int>(nodes + 1 ,q);
  q.memcpy(nidx_d, g.nindex, sizeof(int) * (nodes + 1));
  int *nlist_d = sycl::malloc_device<int>(edges ,q);
  q.memcpy(nlist_d, g.nlist, sizeof(int) * edges);

  int *nlist2_d = sycl::malloc_device<int>(edges ,q);
  int *posscol_d = sycl::malloc_device<int>(nodes ,q);
  int *posscol2_d = sycl::malloc_device<int>(g.edges / WS + 1 ,q);
  int *color_d = sycl::malloc_device<int>(nodes ,q);
  int *wl_d = sycl::malloc_device<int>(nodes ,q);
  int *wlsize_d = sycl::malloc_device<int>(1 ,q);

  const int SMs = q.get_device().get_info<sycl::info::device::max_compute_units>();
  const int mTpSM = q.get_device().get_info<sycl::info::device::max_work_group_size>();
  const int blocks = SMs * mTpSM / ThreadsPerBlock;
  printf("Total number of compute units: %d\n", SMs);
  printf("Maximum resident threads per compute unit: %d\n", mTpSM);
  printf("Work-group size: %d\n", ThreadsPerBlock);
  printf("Total number of work-groups: %d\n", blocks);

  q.wait();

  auto start = std::chrono::high_resolution_clock::now();

  sycl::range<1> gws (blocks * ThreadsPerBlock);
  sycl::range<1> lws (ThreadsPerBlock);

  for (int n = 0; n < repeat; n++) {
    q.memset(wlsize_d, 0, sizeof(int));

    q.submit([&] (sycl::handler &cgh) {
#ifdef SYCL_STREAM
      stream out(64*1024, 256, cgh);
#endif
      cgh.parallel_for<class init_kernel>(sycl::nd_range<1>(gws, lws),
        [=] (sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        init(nodes, edges, nidx_d, nlist_d, nlist2_d,
             posscol_d, posscol2_d, color_d,
             wl_d, wlsize_d, item
#ifdef SYCL_STREAM
             , out
#endif
             );
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class runLarge_kernel>(sycl::nd_range<1>(gws, lws),
        [=] (sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        runLarge(nodes, nidx_d, nlist2_d,
                 posscol_d, posscol2_d, color_d,
                 wl_d, wlsize_d, item);
        });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class runSmall_kernel>(sycl::nd_range<1>(gws, lws), 
        [=] (sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        runSmall(nodes, nidx_d, nlist_d, 
                 posscol_d, color_d, item);
      });
    });
  }

  q.wait();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  double runtime = elapsed_seconds.count() / repeat;

  printf("average runtime: (%d runs):   %.6f s\n", repeat, runtime);
  printf("throughput: %.6f Mnodes/s\n", g.nodes * 0.000001 / runtime);
  printf("throughput: %.6f Medges/s\n", g.edges * 0.000001 / runtime);

  q.memcpy(color, color_d, sizeof(int) * nodes).wait();

  sycl::free(nlist2_d, q);
  sycl::free(posscol_d, q);
  sycl::free(posscol2_d, q);
  sycl::free(color_d, q);
  sycl::free(wl_d, q);
  sycl::free(wlsize_d, q);

  bool ok = true;
  for (int v = 0; v < g.nodes; v++) {
    if (color[v] < 0) {
      printf("ERROR: found unprocessed node in graph (node %d with deg %d)\n\n",
          v, g.nindex[v + 1] - g.nindex[v]);
      ok = false;
      break;
    }
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      if (color[g.nlist[i]] == color[v]) {
        printf("ERROR: found adjacent nodes with same color %d (%d %d)\n\n",
            color[v], v, g.nlist[i]);
        ok = false;
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  if (ok) {
    const int vals = 16;
    int c[vals];
    for (int i = 0; i < vals; i++) c[i] = 0;
    int cols = -1;
    for (int v = 0; v < g.nodes; v++) {
      cols = std::max(cols, color[v]);
      if (color[v] < vals) c[color[v]]++;
    }
    cols++;
    printf("Number of distinct colors used: %d\n", cols);

    int sum = 0;
    for (int i = 0; i < std::min(vals, cols); i++) {
      sum += c[i];
      printf("color %2d: %10d (%5.1f%%)\n", i, c[i], 100.0 * sum / g.nodes);
    }
  }

  delete [] color;
  freeECLgraph(g);
  return 0;
}
