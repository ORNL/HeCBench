#include <complex>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cuda.h>
#include "kernels.h"
#include "reference.h"
#include <cstdlib>

template <typename T>
void init_p(T *p_real, T *p_imag, int width, int height) {
  double s = 64.0;
  for (int j = 1; j <= height; j++) {
    for (int i = 1; i <= width; i++) {
      // p(i,j)=exp(-((i-180.)**2+(j-300.)**2)/(2*s**2))*exp(im*0.4*(i+j-480.))
      std::complex<T> tmp = std::complex<T>(
        exp(-(pow(i - 180.0, 2.0) + pow(j - 300.0, 2.0)) / (2.0 * pow(s, 2.0))), 0.0) *
        exp(std::complex<T>(0.0, 0.4 * (i + j - 480.0)));

      p_real[(j-1) * width + i-1] = real(tmp);
      p_imag[(j-1) * width + i-1] = imag(tmp);
    }
  }
}

template <typename T>
void tsa(int width, int height, int repeat) {

  int numel = width * height;
  size_t matrix_size = numel * sizeof(T);

  T * p_real = new T[numel];
  T * p_imag = new T[numel];
  T * h_real = new T[numel];
  T * h_imag = new T[numel];

  // initialize p_real and p_imag matrices
  init_p(p_real, p_imag, width, height);

  // precomputed values
  T a = cos(0.02);
  T b = sin(0.02);

  // compute reference results
  memcpy(h_imag, p_imag, matrix_size);
  memcpy(h_real, p_real, matrix_size);
  reference(h_real, h_imag, a, b, width, height, repeat);

  // thread block / shared memory block width
  const int BLOCK_X = 16;
  // shared memory block height
  const int BLOCK_Y = sizeof(T) == 8 ? 32 : 96;
  // thread block height
  const int STRIDE_Y = 16;

  // halo sizes on each side
  const int MARGIN_X = 3;
  const int MARGIN_Y = 4;

  // time step
  const int STEPS = 1;

  dim3 grids ((width + (BLOCK_X - 2 * STEPS * MARGIN_X) - 1) / (BLOCK_X - 2 * STEPS * MARGIN_X),
              (height + (BLOCK_Y - 2 * STEPS * MARGIN_Y) - 1) / (BLOCK_Y - 2 * STEPS * MARGIN_Y));
  dim3 blocks (BLOCK_X, STRIDE_Y);
  int sense = 0;

  // pointers to ping-pong arrays
  T *d_real[2];
  T *d_imag[2];

  cudaMalloc((void**)(&d_real[0]), matrix_size);
  cudaMalloc((void**)(&d_real[1]), matrix_size);
  cudaMalloc((void**)(&d_imag[0]), matrix_size);
  cudaMalloc((void**)(&d_imag[1]), matrix_size);
  cudaMemcpy(d_real[0], p_real, matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_imag[0], p_imag, matrix_size, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    kernel<T, STEPS, BLOCK_X, BLOCK_Y, MARGIN_X, MARGIN_Y, STRIDE_Y>
          <<<grids, blocks>>>(a, b, width, height,
           d_real[sense], d_imag[sense], d_real[1-sense], d_imag[1-sense]);
    sense = 1 - sense; // swap
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(p_real, d_real[sense], matrix_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(p_imag, d_imag[sense], matrix_size, cudaMemcpyDeviceToHost);

  // verify
  bool ok = true;
  for (int i = 0; i < numel; i++) {
    if (fabs(p_real[i] - h_real[i]) > 1e-3) {
      ok = false;
      break;
    }
    if (fabs(p_imag[i] - h_imag[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  delete[] p_real;
  delete[] p_imag;
  delete[] h_real;
  delete[] h_imag;
  cudaFree(d_real[0]);
  cudaFree(d_real[1]);
  cudaFree(d_imag[0]);
  cudaFree(d_imag[1]);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <matrix width> <matrix height> <repeat>\n", argv[0]);
    return 1;
  }
  int width = atoi(argv[1]);   // matrix width
  int height = atoi(argv[2]);  // matrix height
  int repeat = atoi(argv[3]);  // repeat kernel execution

  printf("TSA in float32\n");
  tsa<float>(width, height, repeat);

  printf("\n");

  printf("TSA in float64\n");
  tsa<double>(width, height, repeat);
  return 0;
}
