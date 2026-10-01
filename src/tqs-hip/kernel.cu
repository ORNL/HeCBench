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

__global__
void TaskQueue_gpu(const task_t *__restrict__ queue,
                   int *__restrict__ data,
                   int *__restrict__ consumed,
                   const int iterations,
                   const int offset,
                   const int gpuQueueSize)
{
  extern __shared__ int l_mem[];
  int& next = l_mem[0];
  task_t* t = (task_t*)&l_mem[1];

  const int tid       = threadIdx.x;
  const int tile_size = blockDim.x;

  // Fetch task
  if(tid == 0) {
    next = atomicAdd(consumed, 1);
    if(next < gpuQueueSize) {
      t->id = queue[next].id;
      t->op = queue[next].op;
    }
  }
  __syncthreads();
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
    __syncthreads();

    if(tid == 0) {
      next = atomicAdd(consumed, 1);
      // Fetch task
      if(next < gpuQueueSize) {
        t->id = queue[next].id;
        t->op = queue[next].op;
      }
    }
    __syncthreads();
  }
}

void call_TaskQueue_gpu(int blocks,
                        int threads,
                        task_t *queue,
                        int *data,
                        int *consumed, 
                        int iterations,
                        int offset,
                        int gpuQueueSize,
                        int l_mem_size)
{
  dim3 dimGrid(blocks);
  dim3 dimBlock(threads);
  TaskQueue_gpu<<<dimGrid, dimBlock, l_mem_size>>>(
    queue, data, consumed, iterations, offset, gpuQueueSize);
}
