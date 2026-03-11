#include <iostream>
#include <limits>
#include <cmath>
#include <chrono>
#include <sycl/sycl.hpp>

using namespace std;

#define BLOCK_SIZE 16
/**
 * Each element of the product matrix c[i][j] is computed from a unique row and
 * column of the factor matrices, a[i][k] and b[k][j]
 */

// Matrix size constants.
constexpr int m_size = 512 * 8;  // Must be a multiple of 8.
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int K = m_size / 2;

#include "verify.cpp"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int i, j;

  // 2D arrays on host side.
  float(*a_host)[N] = new float[M][N];
  float(*b_host)[K] = new float[N][K];
  // host-side cpu result
  float(*c_host)[K] = new float[M][K];
  // host-side gpu result
  float(*c_back)[K] = new float[M][K];

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      a_host[i][j] = 1.f / N;

  srand(123);
  for (i = 0; i < N; i++)
    for (j = 0; j < K; j++)
      b_host[i][j] = rand() % 256;

  for (j = 0; j < K; j++) {
    float sum = 0;
    for (i = 0; i < N; i++)
      sum += b_host[i][j];
    for (i = 0; i < N; i++)
      b_host[i][j] /= sum;
  }

  // Initialize the device queue with the default selector. The device queue is
  // used to enqueue kernels. It encapsulates all states needed for execution.
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Create buffers for matrices, buffer c is bound with host memory c_back
  float *a = sycl::malloc_device<float>(M*N, q);
  q.memcpy(a, a_host, sizeof(float)*M*N);

  float *b = sycl::malloc_device<float>(N*K, q);
  q.memcpy(b, b_host, sizeof(float)*N*K);

  float *c = sycl::malloc_device<float>(M*K, q);

  cout << "Problem size: c(" << M << "," << K << ") = a(" << M << "," << N
       << ") * b(" << N << "," << K << ")\n";

  auto grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  auto grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  sycl::range<2> gws (grid_rows, grid_cols);
  sycl::range<2> lws (BLOCK_SIZE, BLOCK_SIZE);

  for (int k = 1; k <= 4; k++) {
    printf("Minkowski distance with p = %d\n", k);
    const float p = (float)k;
    const float one_over_p = 1.f / p;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class minkowski>(
          sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> index) {
          int row = index.get_global_id(0);
          int col = index.get_global_id(1);
          if( col < K && row < M) {
            float sum = 0;
            //#pragma unroll (4)
            for (int i = 0; i < N; i++) {
              sum += sycl::powr(sycl::fabs(a[row * N + i] - b[i * K + col]), p);
            }
            c[row * K + col] = sycl::powr(sum, one_over_p);
          }
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    q.memcpy(c_back, c, sizeof(float)*M*K).wait();

    #ifdef VERIFY
    VerifyResult(a_host, b_host, c_host, c_back, p, one_over_p);
    #endif
  }

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  return 0;
}
