#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

#define NUM_THREADS 256

// choose integer type to avoid floating-point rounding errors
typedef int scalar_t;

template <typename T>
void ChannelSumNCHW(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict sum,
    T*__restrict sumsq)
{
  #pragma omp target teams distribute num_teams(C)
  for (int c = 0; c < C; c++) {
    T m_val = 0, v_val = 0;
    #pragma omp parallel for collapse(2) \
    reduction(+:m_val, v_val) num_threads(NUM_THREADS)
    for (int n = 0; n < N; n++) {
      for (int hw = 0; hw < HxW; hw++) {
        const int index = (n * C + c) * HxW + hw;
        m_val += *(X + index);
        v_val += *(X + index) * *(X + index);
      }
    }
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
void ChannelSumNHWC(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict sum,
    T*__restrict sumsq)
{
  #pragma omp target teams distribute num_teams(C)
  for (int c = 0; c < C; c++) {
    T m_val = 0, v_val = 0;
    #pragma omp parallel for reduction(+:m_val, v_val) num_threads(NUM_THREADS)
    for (int i = 0; i < N * HxW; i++) {
      const int index = (i * C + c);
      m_val += *(X + index);
      v_val += *(X + index) * *(X + index);
    }
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

void ComputeChannelSumNCHW (
    const int N,
    const int C,
    const int HxW,
    const scalar_t* X,
    scalar_t* sum,
    scalar_t* sumsq,
    long &time,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    ChannelSumNCHW<scalar_t> (N, C, HxW, X, sum, sumsq);
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void ComputeChannelSumNHWC (
    const int N,
    const int C,
    const int HxW,
    const scalar_t* X,
    scalar_t* sum,
    scalar_t* sumsq,
    long &time,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    ChannelSumNHWC<scalar_t> (N, C, HxW, X, sum, sumsq);
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}


int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int W = atoi(argv[1]);
  const int H = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  long time;

  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {

      printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

      int numel = N * C * W * H; // assume no integer overflow

      size_t data_size_bytes = numel * sizeof(scalar_t);
      size_t output_size_bytes = C * sizeof(scalar_t);

      scalar_t *h_X = (scalar_t*) malloc (data_size_bytes);
      scalar_t *h_sum = (scalar_t*) malloc (output_size_bytes);
      scalar_t *h_sumsq = (scalar_t*) malloc (output_size_bytes);
      scalar_t *r_sum = (scalar_t*) malloc (output_size_bytes);

      srand(numel);
      for (int i = 0; i < numel; i++) h_X[i] = rand() % 256;

      #pragma omp target data map (to: h_X[0:numel]) \
                              map (from: h_sum[0:C], h_sumsq[0:C])
      {
        ComputeChannelSumNHWC (N, C, W*H, h_X, h_sum, h_sumsq, time, repeat);

        #pragma omp target update from (h_sum[0:C])
        ref_nhwc (N, C, W*H, h_X, r_sum, h_sumsq);
        bool ok = check(C, h_sum, r_sum);

        printf("Average time of channel sum (nhwc): %f (ms)\n", (time * 1e-6f) / repeat);
        printf("Verification %s for channel sum (nhwc)\n", ok ? "PASS" : "FAIL");
        if (!ok) exit(1);

        ComputeChannelSumNCHW (N, C, W*H, h_X, h_sum, h_sumsq, time, repeat);

        #pragma omp target update from (h_sum[0:C])
        ref_nchw (N, C, W*H, h_X, r_sum, h_sumsq);
        ok = check(C, h_sum, r_sum);
        
        printf("Average time of channel sum (nchw): %f (ms)\n", (time * 1e-6f) / repeat);
        printf("Verification %s for channel sum (nchw)\n", ok ? "PASS" : "FAIL");
        if (!ok) exit(1);
      }

      free(h_X);
      free(h_sum);
      free(r_sum);
      free(h_sumsq);
    }
  }
  
  end: return 0;
}
