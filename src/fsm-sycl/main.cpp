/* 
   FSM_GA is a GPU-accelerated implementation of a genetic algorithm
   (GA) for finding well-performing finite-state machines (FSM) for predicting
   binary sequences.

   Copyright (c) 2013, Texas State University. All rights reserved.

   Redistribution and use in source and binary forms, with or without modification,
   are permitted for academic, research, experimental, or personal use provided
   that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions, and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions, and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
 contributors may be used to endorse or promote products derived from this
 software without specific prior written permission.

 For all other uses, please contact the Office for Commercialization and Industry
 Relations at Texas State University <http://www.txstate.edu/ocir/>.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Martin Burtscher
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <sycl/sycl.hpp>
#include "parameters.h"
#include "kernels.h"

int main(int argc, char *argv[])
{
  if (argc != 2) {fprintf(stderr, "usage: %s trace_length\n", argv[0]); exit(-1);}
  int length = atoi(argv[1]);

  assert(sizeof(unsigned short) == 2);
  assert(0 < length);
  assert((FSMSIZE & (FSMSIZE - 1)) == 0);
  assert((TABSIZE & (TABSIZE - 1)) == 0);
  assert((0 < FSMSIZE) && (FSMSIZE <= 256));
  assert((0 < TABSIZE) && (TABSIZE <= 32768));
  assert(0 < POPCNT);
  assert((0 < POPSIZE) && (POPSIZE <= 1024));
  assert(0 < CUTOFF);

  int i, j, d, s, bit, pc, misses, besthits, generations;
  unsigned short *data;
  unsigned char state[TABSIZE], fsm[FSMSIZE * 2];
  int best[FSMSIZE * 2 + 3], trans[FSMSIZE][2];
  double runtime;
  struct timeval starttime, endtime;

  data = (unsigned short*) malloc (sizeof(unsigned short) * length);

  srand(123);
  for (int i = 0; i < length; i++) data[i] = rand();

  printf("%d\t#kernel execution times\n", REPEAT);
  printf("%d\t#fsm size\n", FSMSIZE);
  printf("%d\t#entries\n", length);
  printf("%d\t#tab size\n", TABSIZE);
  printf("%d\t#blocks\n", POPCNT);
  printf("%d\t#threads\n", POPSIZE);
  printf("%d\t#cutoff\n", CUTOFF);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  unsigned short *d_data = sycl::malloc_device<unsigned short>(length, q);
  q.memcpy(d_data, data, sizeof(unsigned short) * length);

  int *d_best = sycl::malloc_device<int>(FSMSIZE*2+3, q);
  unsigned int *d_state = sycl::malloc_device<unsigned  int>(POPCNT * POPSIZE, q);
  unsigned char *d_bfsm = sycl::malloc_device<unsigned char>(POPCNT * FSMSIZE *2, q);
  unsigned char *d_same = sycl::malloc_device<unsigned char>(POPCNT, q);
  int *d_smax = sycl::malloc_device<int>(POPCNT, q);
  int *d_sbest = sycl::malloc_device<int>(POPCNT, q);
  int *d_oldmax = sycl::malloc_device<int>(POPCNT, q);

  q.wait();
  gettimeofday(&starttime, NULL);

  for (int i = 0; i < REPEAT; i++) {
    q.memset(d_best, 0, sizeof(int) * (FSMSIZE*2+3));

    sycl::range<1> lws (POPSIZE);
    sycl::range<1> gws (POPCNT * POPSIZE);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<unsigned char, 1> sm (sycl::range<1>(FSMSIZE*2*POPSIZE), cgh);
      cgh.parallel_for<class fsm>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        FSMKernel(item,
                  length,
                  d_data,
                  d_best,
                  d_state,
                  d_bfsm,
                  d_same,
                  d_smax,
                  d_sbest,
                  d_oldmax,
                  sm.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });

    q.submit([&](sycl::handler& cgh) {
      cgh.single_task([=] () {
         MaxKernel(d_best, d_bfsm);
      });
    });
  }
  q.wait();
  gettimeofday(&endtime, NULL);

  runtime = endtime.tv_sec + endtime.tv_usec / 1000000.0 - starttime.tv_sec - starttime.tv_usec / 1000000.0;
  printf("%.6lf\t#runtime [s]\n", runtime / REPEAT);

  q.memcpy(best, d_best, sizeof(int) * (FSMSIZE * 2 + 3)).wait();

  besthits = best[1];
  generations = best[2];
  printf("%.6lf\t#throughput [Gtr/s]\n", 0.000000001 * POPSIZE * generations * length / (runtime / REPEAT));

  // evaluate saturating up/down counter
  for (i = 0; i < FSMSIZE; i++) {
    fsm[i * 2 + 0] = i - 1;
    fsm[i * 2 + 1] = i + 1;
  }
  fsm[0] = 0;
  fsm[(FSMSIZE - 1) * 2 + 1] = FSMSIZE - 1;
  memset(state, 0, TABSIZE);
  misses = 0;
  for (i = 0; i < length; i++) {
    d = (int)data[i];
    pc = (d >> 1) & (TABSIZE - 1);
    bit = d & 1;
    s = (int)state[pc];
    misses += bit ^ (((s + s) / FSMSIZE) & 1);
    state[pc] = fsm[s + s + bit];
  }
  printf("%d\t#sudcnt hits\n", length-misses);
  printf("%d\t#GAfsm hits\n", besthits);

  printf("%.3lf%%\t#sudcnt hits\n", 100.0 * (length - misses) / length);
  printf("%.3lf%%\t#GAfsm hits\n\n", 100.0 * besthits / length);

  // verify result and count transitions
  for (i = 0; i < FSMSIZE; i++) {
    for (j = 0; j < 2; j++) {
      trans[i][j] = 0;
    }
  }
  for (i = 0; i < FSMSIZE * 2; i++) {
    fsm[i] = best[i + 3];
  }
  memset(state, 0, TABSIZE);
  misses = 0;
  for (i = 0; i < length; i++) {
    d = (int)data[i];
    pc = (d >> 1) & (TABSIZE - 1);
    bit = d & 1;
    s = (int)state[pc];
    trans[s][bit]++;
    misses += bit ^ (s & 1);
    state[pc] = (unsigned char)fsm[s + s + bit];
  }

  bool ok = ((length - misses) == besthits);
  printf("%s\n", ok ? "PASS" : "FAIL");
  
#ifdef DEBUG
  // print FSM state assignment in R's ncol format
  for (bit = 0; bit < 2; bit++) {
    for (s = 0; s < FSMSIZE; s++) {
      d = fsm[s + s + bit];
      printf("%c%d %c%d %d\n", (s & 1) ? 'P' : 'N', s / 2, (d & 1) ? 'P' : 'N', d / 2, ((bit * 2) - 1) * trans[s][bit]);
    }
  }
#endif

  free(data);
  sycl::free(d_data, q);
  sycl::free(d_best, q);
  sycl::free(d_state, q);  
  sycl::free(d_bfsm, q);
  sycl::free(d_same, q);
  sycl::free(d_smax, q);
  sycl::free(d_sbest, q);
  sycl::free(d_oldmax, q);
  if (!ok) return 1;
  return 0;
}
