/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//
// This sample demonstrates the use of streams for concurrent execution. It also
// illustrates how to introduce dependencies between CUDA streams with the
// cudaStreamWaitEvent function.
//

// Devices of compute capability 2.0 or higher can overlap the kernels
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

// This is a kernel that does no real work but runs at least for a specified
// number
__global__ void clock_block(long *d_o, long clock_count) {
  long clock_offset = 0;
  for (int i = 0; i < clock_count; i++)
    clock_offset += i % 3;
  d_o[0] = clock_offset;
}

// Single warp reduction kernel
__global__ void sum(long *d_clocks, int N) {
  // Handle to thread block group
  __shared__ long s_clocks[32];

  long my_sum = 0;

  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    my_sum += d_clocks[i];
  }

  s_clocks[threadIdx.x] = my_sum;
  __syncthreads();

  for (int i = 16; i > 0; i /= 2) {
    if (threadIdx.x < i) {
      s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
    }
    __syncthreads();
  }

  d_clocks[0] = s_clocks[0];
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <number of concurrent kernels>\n", argv[0]);
    return 1;
  }
    
  int nkernels = atoi(argv[1]);         // number of concurrent kernels (at least 1)
  int nstreams = nkernels + 1;          // use one more stream than concurrent kernel
  int nbytes = nkernels * sizeof(long); // number of data bytes
  float kernel_time = 20;               // time the kernel should run
  int cuda_device = 0;

  printf("[%s] - Starting...\n", argv[0]);

  long start = get_time();

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, cuda_device);

  // allocate host memory
  long *a;  // pointer to the array data in host memory
  cudaMallocHost((void **)&a, nbytes);

  // allocate device memory
  long *d_a;  // pointers to data and init value in the device memory
  cudaMalloc((void **)&d_a, nbytes);

  // allocate and initialize an array of stream handles
  cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

  for (int i = 0; i < nstreams; i++) {
    cudaStreamCreate(&(streams[i]));
  }

  // the events are used for synchronization only and hence do not need to
  // record timings this also makes events not introduce global sync points when
  // recorded which is critical to get overlap
  cudaEvent_t *kernelEvent;
  kernelEvent = (cudaEvent_t *)malloc(nkernels * sizeof(cudaEvent_t));

  for (int i = 0; i < nkernels; i++) {
    cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming);
  }

  // time execution with nkernels streams
  int clockRate;
  cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);
  long time_clocks = (long)(kernel_time * clockRate);
  printf("time clocks = %ld\n", time_clocks);

  long total_clocks = 0;
  // queue nkernels in separate streams and record when they are done
  for (int i = 0; i < nkernels; ++i) {
    clock_block<<<1, 1, 0, streams[i]>>>(&d_a[i], time_clocks);
    total_clocks += time_clocks;
    cudaEventRecord(kernelEvent[i], streams[i]);

    // make the last stream wait for the kernel event to be recorded
    cudaStreamWaitEvent(streams[nstreams - 1], kernelEvent[i], 0);
  }

  // queue a sum kernel and a copy back to host in the last stream.
  // the commands in this stream get dispatched as soon as all the kernel events
  // have been recorded
  sum<<<1, 32, 0, streams[nstreams - 1]>>>(d_a, nkernels);
  cudaMemcpyAsync(a, d_a, sizeof(long), cudaMemcpyDeviceToHost, streams[nstreams - 1]);

  // at this point the CPU has dispatched all work for the GPU and can continue
  // processing other tasks in parallel

  // wait until the GPU is done
  cudaDeviceSynchronize();

  long end = get_time();
  printf("Measured time for sample = %.3fs\n", (end-start) / 1e6f);

  // check the result
  long sum = 0;
  for (int i = 0; i < time_clocks; i++) sum += i % 3;
  printf("%s\n", a[0] == nkernels * sum ? "PASS" : "FAIL");
  if (a[0] != nkernels * sum) exit(1);

  // release resources
  for (int i = 0; i < nkernels; i++) {
    cudaStreamDestroy(streams[i]);
    cudaEventDestroy(kernelEvent[i]);
  }

  free(streams);
  free(kernelEvent);

  cudaFreeHost(a);
  cudaFree(d_a);

  return 0;
}
