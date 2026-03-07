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

///////////////////////////////////////////////////////////////////////////////
//                            LOMBSCARGLE                                    //
///////////////////////////////////////////////////////////////////////////////

/*
   import cupy as cp
   import matplotlib.pyplot as plt
   First define some input parameters for the signal:
   A = 2.
   w = 1.
   phi = 0.5 * cp.pi
   nin = 10000
   nout = 1000000
   r = cp.random.rand(nin)
   x = cp.linspace(0.01, 10*cp.pi, nin)
   Plot a sine wave for the selected times:
   y = A * cp.sin(w*x+phi)
   Define the array of frequencies for which to compute the periodogram:
   f = cp.linspace(0.01, 10, nout)
   Calculate Lomb-Scargle periodogram:
   pgram = cusignal.lombscargle(x, y, f, normalize=True)
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <cuda.h>

__global__ void       
lombscargle( const int x_shape,
    const int freqs_shape,
    const float *__restrict__ x,
    const float *__restrict__ y,
    const float *__restrict__ freqs,
    float *__restrict__ pgram,
    const float y_dot )
{
  const int tx  = ( blockIdx.x * blockDim.x + threadIdx.x ) ;
  const int stride = ( blockDim.x * gridDim.x ) ;

  for ( int tid = tx; tid < freqs_shape; tid += stride ) {

    float freq = freqs[tid] ;

    float xc = 0;
    float xs = 0;
    float cc = 0;
    float ss = 0;
    float cs = 0;
    float c;
    float s; 

    for ( int j = 0; j < x_shape; j++ ) {
      sincosf( freq * x[j], &s, &c );
      xc += y[j] * c;
      xs += y[j] * s;
      cc += c * c;
      ss += s * s;
      cs += c * s;
    }

    float c_tau;
    float s_tau;
    float tau = atan2f( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) ;
    sincosf( freq * tau, &s_tau, &c_tau );
    float c_tau2 = c_tau * c_tau ;
    float s_tau2 = s_tau * s_tau ;
    float cs_tau = 2.0f * c_tau * s_tau ;

    pgram[tid] = ( 0.5f * ( ( ( c_tau * xc + s_tau * xs ) * ( c_tau * xc + s_tau * xs ) /
            ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +
          ( ( c_tau * xs - s_tau * xc ) * ( c_tau * xs - s_tau * xc ) /
            ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) * y_dot;
  }
}

void lombscargle_cpu( const int x_shape,
    const int freqs_shape,
    const float *__restrict__ x,
    const float *__restrict__ y,
    const float *__restrict__ freqs,
    float *__restrict__ pgram,
    const float y_dot )
{
  for ( int tid = 0; tid < freqs_shape; tid ++) {

    float freq = freqs[tid] ;
    float xc = 0;
    float xs = 0;
    float cc = 0;
    float ss = 0;
    float cs = 0;
    float c;
    float s; 

    for ( int j = 0; j < x_shape; j++ ) {
      sincosf( freq * x[j], &s, &c );
      xc += y[j] * c;
      xs += y[j] * s;
      cc += c * c;
      ss += s * s;
      cs += c * s;
    }

    float c_tau;
    float s_tau;
    float tau = atan2f( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) ;
    sincosf( freq * tau, &s_tau, &c_tau );
    float c_tau2 = c_tau * c_tau ;
    float s_tau2 = s_tau * s_tau ;
    float cs_tau = 2.0f * c_tau * s_tau ;

    pgram[tid] = ( 0.5f * ( ( ( c_tau * xc + s_tau * xs ) * ( c_tau * xc + s_tau * xs ) /
            ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +
          ( ( c_tau * xs - s_tau * xc ) * ( c_tau * xs - s_tau * xc ) /
            ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) * y_dot;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int x_shape = 1000;
  const int freqs_shape = 100000;
  const float A = 2.f;
  const float w = 1.0f;
  const float phi = 1.57f; 

  float* x = (float*) malloc (sizeof(float)*x_shape); 
  float* y = (float*) malloc (sizeof(float)*x_shape); 
  float* f = (float*) malloc (sizeof(float)*freqs_shape); 
  float* p  = (float*) malloc (sizeof(float)*freqs_shape); 
  float* p2 = (float*) malloc (sizeof(float)*freqs_shape); 

  for (int i = 0; i < x_shape; i++)
    x[i] = 0.01f + i*(31.4f - 0.01f)/x_shape;

  for (int i = 0; i < x_shape; i++)
    y[i] = A * sinf(w*x[i]+phi);

  for (int i = 0; i < freqs_shape; i++)
    f[i] = 0.01f + i*(10.f-0.01f)/freqs_shape;

  const float y_dot = 2.0f/1.5f;
  float* d_x; 
  float* d_y; 
  float* d_f; 
  float* d_p; 
  cudaMalloc((void**)&d_x, sizeof(float)*x_shape);
  cudaMalloc((void**)&d_y, sizeof(float)*x_shape);
  cudaMalloc((void**)&d_f, sizeof(float)*freqs_shape);
  cudaMalloc((void**)&d_p, sizeof(float)*freqs_shape);
  cudaMemcpy(d_x, x, sizeof(float)*x_shape, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, sizeof(float)*x_shape, cudaMemcpyHostToDevice);
  cudaMemcpy(d_f, f, sizeof(float)*freqs_shape, cudaMemcpyHostToDevice);

  dim3 grids ((freqs_shape + 255)/256*256);
  dim3 threads (256);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++)
    lombscargle<<<grids, threads>>>(x_shape, freqs_shape, d_x, d_y, d_f, d_p, y_dot);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3) / repeat);

  cudaMemcpy(p, d_p, sizeof(float)*freqs_shape, cudaMemcpyDeviceToHost);

  // verification
  lombscargle_cpu(x_shape, freqs_shape, x, y, f, p2, y_dot);

  bool error = false;
  for (int i = 0; i < freqs_shape; i++) {
    if (fabsf(p[i]-p2[i]) > 1e-1f) {
      printf("%.3f %.3f\n", p[i], p2[i]);
      error = true;
      break;
    }
  }

  printf("%s\n", error ? "FAIL" : "PASS");

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_f);
  cudaFree(d_p);
  free(x);
  free(y);
  free(f);
  free(p);
  free(p2);
  if (error) return 1;
  return 0;
}
