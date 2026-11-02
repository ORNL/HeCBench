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

void call_TaskQueue_gpu(int blocks,
                        int threads,
                        const task_t *__restrict task_queue,
                        int *__restrict data_queue,
                        int *__restrict consumed, 
                        int iterations,
                        int offset,
                        int gpuQueueSize)
{
  #pragma omp target teams num_teams(blocks) thread_limit(threads)
  {
    int l_mem[3];
    #pragma omp parallel 
    {
      int& next = l_mem[0];
      task_t* t = (task_t*)&l_mem[1];

      const int tid       = omp_get_thread_num();
      const int tile_size = omp_get_num_threads();

      // Fetch task
      if(tid == 0) {
        #pragma omp atomic capture
        next = (*consumed)++;
        if(next < gpuQueueSize) {
          t->id = task_queue[next].id;
          t->op = task_queue[next].op;
        }
      }

      #pragma omp barrier

      while(next < gpuQueueSize) {
        // Compute task
        if(t->op == SIGNAL_WORK_KERNEL) {
          for(int i = 0; i < iterations; i++) {
            data_queue[(t->id - offset) * tile_size + tid] += tile_size;
          }

          data_queue[(t->id - offset) * tile_size + tid] += t->id;
        }
        if(t->op == SIGNAL_NOTWORK_KERNEL) {
          for(int i = 0; i < 1; i++) {
            data_queue[(t->id - offset) * tile_size + tid] += tile_size;
          }

          data_queue[(t->id - offset) * tile_size + tid] += t->id;
        }
        #pragma omp barrier

        if(tid == 0) {
          #pragma omp atomic capture
          next = (*consumed)++;
          // Fetch task
          if(next < gpuQueueSize) {
            t->id = task_queue[next].id;
            t->op = task_queue[next].op;
          }
        }
        #pragma omp barrier
      }
    }
  }
}
