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

#include "kernel.h"

void TaskQueue_gpu(const task_t *__restrict queue,
                   int *__restrict data,
                   int *__restrict consumed,
                   const int iterations,
                   const int offset,
                   const int gpuQueueSize,
                   sycl::nd_item<1> &item,
                   int *l_mem)
{
  int& next = l_mem[0];
  task_t* t = (task_t*)&l_mem[1];

  const int tid = item.get_local_id(0);
  const int tile_size = item.get_local_range(0);

  // Fetch task
  if(tid == 0) {
    auto ao = sycl::atomic_ref<int,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space> (*consumed);
    next = ao.fetch_add(1);
    if(next < gpuQueueSize) {
      t->id = queue[next].id;
      t->op = queue[next].op;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);
  while(next < gpuQueueSize) {
    // Compute task
    if(t->op == SIGNAL_WORK_KERNEL) {
      for(int i = 0; i < iterations; i++) {
        data[(t->id - offset) * tile_size + tid] += tile_size;
      }

      data[(t->id - offset) * tile_size + tid] += t->id;
    }
    if(t->op == SIGNAL_NOTWORK_KERNEL) {
      for(int i = 0; i < 1; i++) {
        data[(t->id - offset) * tile_size + tid] += tile_size;
      }

      data[(t->id - offset) * tile_size + tid] += t->id;
    }
    item.barrier(sycl::access::fence_space::local_space);

    if(tid == 0) {
      auto ao = sycl::atomic_ref<int,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> (*consumed);
      next = ao.fetch_add(1);
      // Fetch task
      if(next < gpuQueueSize) {
        t->id = queue[next].id;
        t->op = queue[next].op;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
}

void call_TaskQueue_gpu(sycl::queue &q,
                        int blocks,
                        int threads,
                        const task_t *queue,
                        int *data,
                        int *consumed,
                        int iterations,
                        int offset,
                        int gpuQueueSize,
                        int l_mem_size)
{
  sycl::range<1> gws (threads * blocks);
  sycl::range<1> lws (threads);

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int, 1> l_mem (sycl::range<1>(l_mem_size/sizeof(int)), cgh);
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      TaskQueue_gpu(queue, data, consumed, iterations, offset,
                    gpuQueueSize, item, l_mem.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
}
