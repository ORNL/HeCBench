#include <complex>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <sycl/sycl.hpp>
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
void tsa(sycl::queue &q, int width, int height, int repeat) {

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


  sycl::range<3> gws (1, (height + (BLOCK_Y - 2 * STEPS * MARGIN_Y) - 1) /
                      (BLOCK_Y - 2 * STEPS * MARGIN_Y) * STRIDE_Y,
                      (width + (BLOCK_X - 2 * STEPS * MARGIN_X) - 1) /
                      (BLOCK_X - 2 * STEPS * MARGIN_X)  * BLOCK_X);

  sycl::range<3> lws (1, STRIDE_Y, BLOCK_X);
  int sense = 0;

  // pointers to ping-pong arrays
  T *d_real[2];
  T *d_imag[2];

  d_real[0] = sycl::malloc_device<T>(numel, q);
  d_real[1] = sycl::malloc_device<T>(numel, q);
  d_imag[0] = sycl::malloc_device<T>(numel, q);
  d_imag[1] = sycl::malloc_device<T>(numel, q);
  q.memcpy(d_real[0], p_real, matrix_size);
  q.memcpy(d_imag[0], p_imag, matrix_size);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T[BLOCK_Y][BLOCK_X], 0> rl_acc(cgh);
      sycl::local_accessor<T[BLOCK_Y][BLOCK_X], 0> im_acc(cgh);

      auto d_real_ping = d_real[sense];
      auto d_real_pong = d_real[1 - sense];
      auto d_imag_ping = d_imag[sense];
      auto d_imag_pong = d_imag[1 - sense];

      cgh.parallel_for(
          sycl::nd_range<3>(gws, lws),
          [=](sycl::nd_item<3> item) {
            kernel<T, STEPS, BLOCK_X, BLOCK_Y, MARGIN_X, MARGIN_Y, STRIDE_Y>(
                a, b, width, height, d_real_ping, d_imag_ping,
                d_real_pong, d_imag_pong, item, rl_acc, im_acc);
      });
    });
    sense = 1 - sense; // swap
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(p_real, d_real[sense], matrix_size).wait();
  q.memcpy(p_imag, d_imag[sense], matrix_size).wait();

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

  delete[] p_real;
  delete[] p_imag;
  delete[] h_real;
  delete[] h_imag;
  sycl::free(d_real[0], q);
  sycl::free(d_real[1], q);
  sycl::free(d_imag[0], q);
  sycl::free(d_imag[1], q);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <matrix width> <matrix height> <repeat>\n", argv[0]);
    return 1;
  }
  int width = atoi(argv[1]);   // matrix width
  int height = atoi(argv[2]);  // matrix height
  int repeat = atoi(argv[3]);  // repeat kernel execution

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("TSA in float32\n");
  tsa<float>(q, width, height, repeat);

  printf("\n");

  printf("TSA in float64\n");
  tsa<double>(q, width, height, repeat);
  if (!ok) return 1;
  return 0;
}
