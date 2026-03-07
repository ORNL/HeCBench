// Copyright (c) 2019-2020, NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "reference.h"

// CUDA error checking
void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

///////////////////////////////////////////////////////////////////////////////
//                                SOSFILT                                    //
///////////////////////////////////////////////////////////////////////////////
#define MAX_THREADS 256
#define THREADS 64

template<typename T>
__global__ void sosfilt(
    const int n_signals,
    const int n_samples,
    const int n_sections,
    const int zi_width,
    const T *__restrict__ sos,
    const T *__restrict__ zi,
          T *__restrict__ x_in)
{
  extern __shared__ char smem[];
  T *s_out = reinterpret_cast<T *>( smem );
  T *s_zi = reinterpret_cast<T *>( &s_out[n_sections] ) ;
  T *s_sos = reinterpret_cast<T *>( &s_zi[n_sections * zi_width] ) ;

  const int tx = threadIdx.x;
  const int ty = blockIdx.x;

  // Reset shared memory
  s_out[tx] = 0;

  // Load zi
  for ( int i = 0; i < zi_width; i++ ) {
    s_zi[tx * zi_width + i] = zi[ty * n_sections * zi_width + tx * zi_width + i];
  }

  // Load SOS
#pragma unroll
  for ( int i = 0; i < sos_width; i++ ) {
    s_sos[tx * sos_width + i] = sos[tx * sos_width + i];
  }

  __syncthreads( );

  const int load_size = n_sections - 1 ;
  const int unload_size = n_samples - load_size ;

  T temp;
  T x_n;

  if ( ty < n_signals ) {
    // Loading phase
    for ( int n = 0; n < load_size; n++ ) {
      if ( tx == 0 ) {
        x_n = x_in[ty * n_samples + n];
      } else {
        x_n = s_out[tx - 1];
      }

      // Use direct II transposed structure
      temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

      s_zi[tx * zi_width + 0] =
        s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

      s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

      s_out[tx] = temp;

      __syncthreads( );
    }

    // Processing phase
    for ( int n = load_size; n < n_samples; n++ ) {
      if ( tx == 0 ) {
        x_n = x_in[ty * n_samples + n];
      } else {
        x_n = s_out[tx - 1];
      }

      // Use direct II transposed structure
      temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

      s_zi[tx * zi_width + 0] =
        s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

      s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

      if ( tx < load_size ) {
        s_out[tx] = temp;
      } else {
        x_in[ty * n_samples + ( n - load_size )] = temp;
      }

      __syncthreads( );
    }

    // Unloading phase
    for ( int n = 0; n < n_sections; n++ ) {
      // retire threads that are less than n
      if ( tx > n ) {
        x_n = s_out[tx - 1];

        // Use direct II transposed structure
        temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

        s_zi[tx * zi_width + 0] =
          s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

        s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

        if ( tx < load_size ) {
          s_out[tx] = temp;
        } else {
          x_in[ty * n_samples + ( n + unload_size )] = temp;
        }
      }
      __syncthreads( );
    }
  }
}

template <typename T>
void filtering (const int repeat,
                const int n_signals, const int n_samples,
                const int n_sections, const int zi_width)
{
  // the number of second-order sections must be less than max threads per block
  assert(MAX_THREADS >= n_sections);

  // The number of samples must be greater than or equal to the number of sections
  assert(n_samples >= n_sections);

  // randomize input data
  srand(2);

  dim3 blocksPerGrid (n_signals);
  dim3 threadsPerBlock (THREADS);

  // Second-order section digital filter
  const int sos_size = n_sections * sos_width ;

  T* sos = (T*) malloc (sizeof(T) * sos_size);
  for (int i = 0; i < n_sections; i++)
    for (int j = 0; j < sos_width; j++)
      sos[i*sos_width+j] = (T)1 ; // for test

  T* d_sos;
  cudaCheck(cudaMalloc((void**)&d_sos, sizeof(T) * sos_size));
  cudaCheck(cudaMemcpy(d_sos, sos, sizeof(T) * sos_size, cudaMemcpyHostToDevice));

  // initial  conditions
  const int z_size = n_sections * n_signals * zi_width;
  T* zi = (T*) malloc (sizeof(T) * z_size);
  for (int i = 0; i < z_size; i++) zi[i] = (T)1; // for test

  T* d_zi;
  cudaCheck(cudaMalloc((void**)&d_zi, sizeof(T) * z_size));
  cudaCheck(cudaMemcpy(d_zi, zi, sizeof(T) * z_size, cudaMemcpyHostToDevice));

  // input signals
  const int x_size = n_signals * n_samples;
  T* x = (T*) malloc (sizeof(T) * x_size);
  T* x_ref = (T*) malloc (sizeof(T) * x_size);
  for (int i = 0; i < n_signals; i++)
    for (int j = 0; j < n_samples; j++)
      x_ref[i*n_samples+j] = x[i*n_samples+j] = (T)std::sin(2*3.14*(i+1+j));

  T* d_x;
  cudaCheck(cudaMalloc((void**)&d_x, sizeof(T) * x_size));
  cudaCheck(cudaMemcpy(d_x, x, sizeof(T) * x_size, cudaMemcpyHostToDevice));

  const int out_size = n_sections;
  const int shared_mem = (out_size + z_size + sos_size) * sizeof(T);

  // warmup and validate
  for (int n = 0; n < 30; n++) {
    sosfilt<T><<<blocksPerGrid, threadsPerBlock, shared_mem, 0>>>(
      n_signals,
      n_samples,
      n_sections,
      zi_width,
      d_sos,
      d_zi,
      d_x);

    reference<T>(
        n_signals,
        n_samples,
        n_sections,
        zi_width,
        sos,
        zi,
        x_ref);
  }

  cudaCheck(cudaMemcpy(x, d_x, sizeof(T) * x_size, cudaMemcpyDeviceToHost));

  bool ok = compare_results<T>(x_ref, x, n_signals * n_samples, 1e-4, 1e-4);
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  cudaCheck(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    sosfilt<T><<<blocksPerGrid, threadsPerBlock, shared_mem, 0>>>(
      n_signals,
      n_samples,
      n_sections,
      zi_width,
      d_sos,
      d_zi,
      d_x);
  }

  cudaCheck(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %lf (s)\n", time * 1e-9 / repeat);

  free(x);
  free(x_ref);
  free(sos);
  free(zi);
  cudaCheck(cudaFree(d_x));
  cudaCheck(cudaFree(d_sos));
  cudaCheck(cudaFree(d_zi));
}

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int numSections = THREADS;

#ifdef DEBUG
  const int numSignals = 2;
  const int numSamples = THREADS+1;
#else
  // shared memory size depends on numSignals, so it may cause kernel launch failure
  const int numSignals = 8;
  const int numSamples = 1000000;
#endif

  const int zi_width = 2;

  printf("Single-precision second-order-section filtering of digital signals\n");
  filtering<float> (repeat, numSignals, numSamples, numSections, zi_width);

  printf("Double-precision second-order-section filtering of digital signals\n");
  filtering<double> (repeat, numSignals, numSamples, numSections, zi_width);
  return 0;
}
