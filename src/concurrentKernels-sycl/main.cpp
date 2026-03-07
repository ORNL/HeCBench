/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of NVIDIA CORPORATION nor the names of its
     contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.
 
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <sycl/sycl.hpp>


long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

// This is a kernel that does no real work but runs at least for a specified
// number
void clock_block(long *d_o, long clock_count) {
  long clock_offset = 0;
  for (int i = 0; i < clock_count; i++)
    clock_offset += i % 3;
  d_o[0] = clock_offset;
}

// Single warp reduction kernel
void sum(long *d_clocks, int N, sycl::nd_item<1> &item, long *s_clocks) {
  long my_sum = 0;
  int lid = item.get_local_id(0);

  for (int i = lid; i < N; i += item.get_local_range(0)) {
    my_sum += d_clocks[i];
  }

  s_clocks[lid] = my_sum;

  item.barrier(sycl::access::fence_space::local_space);

  for (int i = 16; i > 0; i /= 2) {
    if (lid < i) {
      s_clocks[lid] += s_clocks[lid + i];
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  d_clocks[0] = s_clocks[0];
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <number of concurrent kernels>\n", argv[0]);
    return 1;
  }
    
  int nkernels = atoi(argv[1]);         // number of concurrent kernels (at least 1)
  int nbytes = nkernels * sizeof(long); // number of data bytes
  float kernel_time = 20;               // time the kernel should run

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("[%s] - Starting...\n", argv[0]);

  long start = get_time();

  // kernel events
  std::vector<sycl::event> e (nkernels);

  // allocate host memory
  long *a = (long *)sycl::malloc_host(nbytes, q);

  // allocate device memory
  long *d_a = (long *)sycl::malloc_device(nbytes, q);

  // time execution with nkernels streams
  unsigned clock_rate = 1e3 * q.get_device().get_info<sycl::info::device::max_clock_frequency>();
  long time_clocks = (long)(kernel_time * clock_rate);

  printf("time clocks = %ld\n", time_clocks);

  // queue nkernels with events recorded
  sycl::range<1> gws (1);
  sycl::range<1> lws (1);
  for (int i = 0; i < nkernels; ++i) {
    e[i] = q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        clock_block(d_a+i, time_clocks);
      });
    });
  }

  // queue a sum kernel and a copy back to host 
  sycl::range<1> gws2 (32);
  sycl::range<1> lws2 (32);
  auto e2 = q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<long, 1> s (sycl::range<1>(32), cgh);
    cgh.depends_on(e);
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      sum(d_a, nkernels, item, s.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });

  // at this point the CPU has dispatched all work for the GPU and can continue
  // processing other tasks in parallel

  // wait until the GPU is done
  q.memcpy(a, d_a, sizeof(long), e2).wait();

  long end = get_time();
  printf("Measured time for sample = %.3fs\n", (end-start) / 1e6f);

  // check the result
  long sum = 0;
  for (int i = 0; i < time_clocks; i++) sum += i % 3;
  printf("%s\n", a[0] == nkernels * sum ? "PASS" : "FAIL");
  if (a[0] != nkernels * sum) exit(1);

  sycl::free(a, q);
  sycl::free(d_a, q);

  return 0;
}
