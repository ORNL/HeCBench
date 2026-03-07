/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda_runtime.h>

__global__
void incKernel(int *g_out, const int *g_in, int N, int inner_reps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    for (int i = 0; i < inner_reps; ++i) {
      g_out[idx] = (i == 0 ? g_in[idx] : g_out[idx]) + 1;
    }
  }
}

#define STREAM_COUNT 4

int *h_data_in[STREAM_COUNT];
int *d_data_in[STREAM_COUNT];

int *h_data_out[STREAM_COUNT];
int *d_data_out[STREAM_COUNT];

cudaStream_t stream[STREAM_COUNT];

int N = 1 << 22;
int nreps = 10;  // number of times each experiment is repeated
int inner_reps = 5;  // loop iterations in the GPU kernel

int memsize;

dim3 block (256, 1, 1);
dim3 grid (N/256, 1, 1);

float processWithStreams(int streams_used);
bool check();


int main(int argc, char *argv[]) {

  printf("Length of the array = %d\n", N);

  memsize = N * sizeof(int);

  // Allocate resources
  for (int i = 0; i < STREAM_COUNT; ++i) {
    cudaHostAlloc(&h_data_in[i], memsize, cudaHostAllocDefault);
    cudaHostAlloc(&h_data_out[i], memsize, cudaHostAllocDefault);

    cudaMalloc(&d_data_in[i], memsize);
    cudaMemset(d_data_in[i], 0, memsize);

    cudaMalloc(&d_data_out[i], memsize);

    cudaStreamCreate(&stream[i]);
  }

  // initialize host memory
  for (int i = 0; i < STREAM_COUNT; ++i) {
    memset(h_data_in[i], 0, memsize);
  }

  // Process pipelined work
  float serial_time = processWithStreams(1);
  float overlap_time = processWithStreams(STREAM_COUNT);

  printf("\nAverage measured timings over %d repetitions:\n", nreps);
  printf(" Avg. time when execution fully serialized\t: %f ms\n",
         serial_time / nreps);
  printf(" Avg. time when overlapped using %d streams\t: %f ms\n", STREAM_COUNT,
         overlap_time / nreps);
  printf(" Avg. speedup gained (serialized - overlapped)\t: %f\n",
         (serial_time - overlap_time) / nreps);

  printf("\nMeasured throughput:\n");
  printf(" Fully serialized execution\t\t: %f GB/s\n",
         (nreps * (memsize * 2e-6)) / serial_time);
  printf(" Overlapped using %d streams\t\t: %f GB/s\n", STREAM_COUNT,
         (nreps * (memsize * 2e-6)) / overlap_time);

  // Verify the results, we will use the results for final output
  bool bResults = check();
  printf("\n%s\n", bResults ? "PASS" : "FAIL");
  if (!bResults) exit(1);

  // Free resources
  for (int i = 0; i < STREAM_COUNT; ++i) {
    cudaFreeHost(h_data_in[i]);
    cudaFree(d_data_in[i]);

    cudaFreeHost(h_data_out[i]);
    cudaFree(d_data_out[i]);

    cudaStreamDestroy(stream[i]);
  }

  // Test result
  exit(bResults ? EXIT_SUCCESS : EXIT_FAILURE);
}

float processWithStreams(int streams_used) {
  int current_stream = 0;

  auto start = std::chrono::steady_clock::now();

  // Do processing in a loop
  //
  // Note: All memory commands are processed in the order they are issued,
  // independent of the stream they are enqueued in. Hence the pattern by
  // which the copy and kernel commands are enqueued in the stream
  // has an influence on the achieved overlap.

  for (int i = 0; i < nreps; ++i) {
    int next_stream = (current_stream + 1) % streams_used;

    // Process current frame
    incKernel<<<grid, block, 0, stream[current_stream]>>>(
        d_data_out[current_stream], d_data_in[current_stream], N, inner_reps);

    // Upload next frame
    cudaMemcpyAsync(d_data_in[next_stream], h_data_in[next_stream], memsize,
                    cudaMemcpyHostToDevice, stream[next_stream]);

    // Download current frame
    cudaMemcpyAsync(h_data_out[current_stream], d_data_out[current_stream], memsize,
                    cudaMemcpyDeviceToHost, stream[current_stream]);

    current_stream = next_stream;
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return (time * 1e-6f); // milliseconds
}

bool check() {
  bool passed = true;

  for (int j = 0; j < STREAM_COUNT; ++j) {
    for (int i = 0; i < N; ++i) {
      passed &= (h_data_out[j][i] == inner_reps);
    }
  }
  return passed;
}
