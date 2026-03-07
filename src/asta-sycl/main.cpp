/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <algorithm>  // for_each
#include <chrono>
#include <sycl/sycl.hpp>

#include "support/common.h"
#include "support/verify.h"
#include <cstdlib>


// Params ---------------------------------------------------------------------
struct Params {

  int device;
  int n_gpu_threads;
  int n_gpu_blocks;
  int n_warmup;
  int n_reps;
  int m;
  int n;
  int s;

  Params(int argc, char **argv) {
    device        = 0;
    n_gpu_threads = 64;
    n_gpu_blocks  = 16;
    n_warmup      = 10;
    n_reps        = 100;
    m             = 197;
    n             = 35588;
    s             = 32;
    int opt;
    while((opt = getopt(argc, argv, "hd:i:g:w:r:m:n:s:")) >= 0) {
      switch(opt) {
        case 'h':
          usage();
          exit(0);
          break;
        case 'i': n_gpu_threads = atoi(optarg); break;
        case 'g': n_gpu_blocks  = atoi(optarg); break;
        case 'w': n_warmup      = atoi(optarg); break;
        case 'r': n_reps        = atoi(optarg); break;
        case 'm': m             = atoi(optarg); break;
        case 'n': n             = atoi(optarg); break;
        case 's': s             = atoi(optarg); break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
      }
    }
  }

  void usage() {
    fprintf(stderr,
        "\nUsage:  ./main [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -i <I>    # of device threads per block (default=64)"
        "\n    -g <G>    # of device blocks (default=16)"
        "\n    -w <W>    # of warmup iterations (default=10)"
        "\n    -r <R>    # of repetition iterations (default=100)"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -m <M>    matrix height (default=197)"
        "\n    -n <N>    matrix width (default=35588)"
        "\n    -s <M>    super-element size (default=32)"
        "\n");
  }
};

// Input Data -----------------------------------------------------------------
void read_input(T *x_vector, const Params &p) {
  int tiled_n = divceil(p.n, p.s);
  int in_size = p.m * tiled_n * p.s;
  srand(5432);
  for(int i = 0; i < in_size; i++) {
    x_vector[i] = ((T)(rand() % 100) / 100);
  }
}

inline int atomicAdd(int& val, const int delta)
{
  sycl::atomic_ref<int, sycl::memory_order::relaxed, 
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ref(val);
  return ref.fetch_add(delta);
}

inline int atomicExch(int& addr, const int val) {
  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ref(addr);
  return ref.exchange(val);
}

int main(int argc, char **argv) {

  const Params p(argc, argv);
  int blocks = p.n_gpu_blocks;
  int threads = p.n_gpu_threads;
  const int max_gpu_threads = 256;
  assert(threads <= max_gpu_threads && 
         "The thread block size is at most 256");

  // Allocate
  int tiled_n       = divceil(p.n, p.s);
  int in_size       = p.m * tiled_n * p.s;
  int finished_size = p.m * tiled_n;
  
  size_t in_size_bytes = in_size * sizeof(T);
  size_t finished_size_bytes = finished_size * sizeof(int);

  T *h_in_out = (T *)malloc(in_size_bytes);
  int *h_finished = (int *)malloc(finished_size_bytes);
  int *h_head = (int *)malloc(sizeof(int));

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    T *d_in_out = sycl::malloc_device<T>(in_size, q);
  int *d_finished = sycl::malloc_device<int>(finished_size, q);
  int *d_head = sycl::malloc_device<int>(1, q);

  T *h_in_backup = (T *)malloc(in_size_bytes);

  // Initialize
  read_input(h_in_out, p);
  memset((void *)h_finished, 0, finished_size_bytes);
  h_head[0] = 0;
  memcpy(h_in_backup, h_in_out, in_size_bytes); // Backup for reuse across iterations

  const int A = p.m;
  const int B = tiled_n;
  const int b = p.s;

  double time = 0; 

  sycl::range<1> gws (blocks*threads);
  sycl::range<1> lws (threads);

  // Loop over the kernel on a device
  for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
    q.memcpy(d_in_out, h_in_backup, in_size_bytes);
    q.memcpy(d_finished, h_finished, finished_size_bytes);
    q.memcpy(d_head, h_head, sizeof(int));

    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<int, 1> lmem(sycl::range<1>(2), cgh);
      cgh.parallel_for<class asta>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {

        const int tid = item.get_local_id(0);
        int       m   = A * B - 1;

        if(tid == 0) // Dynamic fetch
          lmem[1] = atomicAdd(d_head[0], 1);
        item.barrier(sycl::access::fence_space::local_space);

        while(lmem[1] < m) {
          int next_in_cycle = (lmem[1] * A) - m * (lmem[1] / B);
          if(next_in_cycle == lmem[1]) {
            if(tid == 0) // Dynamic fetch
              lmem[1] = atomicAdd(d_head[0], 1);
            item.barrier(sycl::access::fence_space::local_space);
            continue;
          }
          T   data1, data2, data3, data4;
          int i = tid;
          if(i < b)
            data1 = d_in_out[lmem[1] * b + i];
          i += item.get_local_range(0);
          if(i < b)
            data2 = d_in_out[lmem[1] * b + i];
          i += item.get_local_range(0);
          if(i < b)
            data3 = d_in_out[lmem[1] * b + i];
          i += item.get_local_range(0);
          if(i < b)
            data4 = d_in_out[lmem[1] * b + i];

          if(tid == 0) {
            //make sure the read is not cached
            lmem[0] = atomicAdd(d_finished[lmem[1]], 0);
          }
          item.barrier(sycl::access::fence_space::local_space);

          for(; lmem[0] == 0; next_in_cycle = (next_in_cycle * A) - m * (next_in_cycle / B)) {
            T backup1, backup2, backup3, backup4;
            i = tid;
            if(i < b)
              backup1 = d_in_out[next_in_cycle * b + i];
            i += item.get_local_range(0);
            if(i < b)
              backup2 = d_in_out[next_in_cycle * b + i];
            i += item.get_local_range(0);
            if(i < b)
              backup3 = d_in_out[next_in_cycle * b + i];
            i += item.get_local_range(0);
            if(i < b)
              backup4 = d_in_out[next_in_cycle * b + i];

            if(tid == 0) {
              lmem[0] = atomicExch(d_finished[next_in_cycle], (int)1);
            }
            item.barrier(sycl::access::fence_space::local_space);

            if(!lmem[0]) {
              i = tid;
              if(i < b)
                d_in_out[next_in_cycle * b + i] = data1;
              i += item.get_local_range(0);
              if(i < b)
                d_in_out[next_in_cycle * b + i] = data2;
              i += item.get_local_range(0);
              if(i < b)
                d_in_out[next_in_cycle * b + i] = data3;
              i += item.get_local_range(0);
              if(i < b)
                d_in_out[next_in_cycle * b + i] = data4;
            }
            i = tid;
            if(i < b)
              data1 = backup1;
            i += item.get_local_range(0);
            if(i < b)
              data2 = backup2;
            i += item.get_local_range(0);
            if(i < b)
              data3 = backup3;
            i += item.get_local_range(0);
            if(i < b)
              data4 = backup4;
          }

          if(tid == 0) // Dynamic fetch
            lmem[1] = atomicAdd(d_head[0], 1);
          item.barrier(sycl::access::fence_space::local_space);
        }
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();

    if (rep >= p.n_warmup) 
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    q.memcpy(h_in_out, d_in_out, in_size_bytes).wait();
  }

  printf("Average kernel execution time %lf (s)\n", (time * 1e-9) / p.n_reps);

  // Verify
  int status = verify(h_in_out, h_in_backup, tiled_n * p.s, p.m, p.s);
  printf("%s\n", (status == 0) ? "PASS" : "FAIL");

  // Free memory
  free(h_in_out);
  free(h_finished);
  free(h_head);
  free(h_in_backup);

  sycl::free(d_in_out, q);
  sycl::free(d_finished, q);
  sycl::free(d_head, q);
  if (!(status == 0)) return 1;
  return 0;
}
