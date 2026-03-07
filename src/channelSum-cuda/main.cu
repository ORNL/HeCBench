#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include <cub/cub.cuh>
#include "reference.h"

#define NUM_THREADS 256

// choose integer type to avoid floating-point rounding errors
typedef int scalar_t;

#define DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE(                      \
    size, Func, T, grid_dim, stream, ...)                                  \
  do {                                                                     \
    if (size >= 128) {                                                     \
      Func<T, 1, 128><<<grid_dim, dim3(1, 128), 0, stream>>>(__VA_ARGS__); \
    } else if (size >= 64) {                                               \
      Func<T, 2, 64><<<grid_dim, dim3(2, 64), 0, stream>>>(__VA_ARGS__);   \
    } else if (size >= 32) {                                               \
      Func<T, 4, 32><<<grid_dim, dim3(4, 32), 0, stream>>>(__VA_ARGS__);   \
    } else {                                                               \
      Func<T, 8, 16><<<grid_dim, dim3(8, 16), 0, stream>>>(__VA_ARGS__);   \
    }                                                                      \
  } while (false)


template <typename T, int kBlockDimX, int kBlockDimY>
__device__ void BlockReduce(T &input1, T &input2) {
  using BlockReduce = cub::BlockReduce<T,
                                       kBlockDimX,
                                       cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                       kBlockDimY>;
  __shared__ typename BlockReduce::TempStorage temp_storage1;
  __shared__ typename BlockReduce::TempStorage temp_storage2;
  input1 = BlockReduce(temp_storage1).Sum(input1);
  input2 = BlockReduce(temp_storage2).Sum(input2);
}

template <typename T>
__device__ void BlockReduce(T &input1, T &input2) {
  using BlockReduce = cub::BlockReduce<T, NUM_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage1;
  __shared__ typename BlockReduce::TempStorage temp_storage2;
  input1 = BlockReduce(temp_storage1).Sum(input1);
  input2 = BlockReduce(temp_storage2).Sum(input2);
}

template <typename T, int kBlockDimX, int kBlockDimY>
__global__
void ChannelSumNCHW(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{

  T m_val = 0;
  T v_val = 0;
  const int c = blockIdx.x;

  // sum batches from different channels
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    for (int hw = threadIdx.y; hw < HxW; hw += blockDim.y) {
      const int index = (n * C + c) * HxW + hw;
      m_val += __ldg(X + index);
      v_val += __ldg(X + index) * __ldg(X + index);
    }
  }
  BlockReduce<T, kBlockDimX, kBlockDimY>(m_val, v_val);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
__global__
void ChannelSumNHWC(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  const int inner_size = N * HxW;
  const int c = blockIdx.x;
  T m_val = 0;
  T v_val = 0;
  for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
    const int index = i * C + c;
    m_val += __ldg(X + index);
    v_val += __ldg(X + index) * __ldg(X + index);
  }
  BlockReduce<T>(m_val, v_val);
  if (threadIdx.x == 0) {
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
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE(
      HxW,
      ChannelSumNCHW,
      scalar_t,
      C,
      0,
      N,
      C,
      HxW,
      X,
      sum,
      sumsq);
  }

  cudaDeviceSynchronize();
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
    ChannelSumNHWC<scalar_t> <<<C, NUM_THREADS>>>(N, C, HxW, X, sum, sumsq);
  }

  cudaDeviceSynchronize();
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

      scalar_t *d_X, *d_sum, *d_sumsq;
      auto errorX = cudaMalloc((void**)&d_X, data_size_bytes);
      auto errorY = cudaMalloc((void**)&d_sum, output_size_bytes);
      auto errorZ = cudaMalloc((void**)&d_sumsq, output_size_bytes);
      if (errorX != cudaSuccess ||
          errorY != cudaSuccess ||
          errorZ != cudaSuccess) {
        if (errorX == cudaSuccess) cudaFree(d_X);
        if (errorY == cudaSuccess) cudaFree(d_sum);
        if (errorZ == cudaSuccess) cudaFree(d_sumsq);
        printf("Device memory allocation failed. Exit\n");
        goto end;
      }

      cudaMemcpy(d_X, h_X, data_size_bytes, cudaMemcpyHostToDevice);

      ComputeChannelSumNHWC (N, C, W*H, d_X, d_sum, d_sumsq, time, repeat);

      cudaMemcpy(h_sum, d_sum, output_size_bytes, cudaMemcpyDeviceToHost);
      ref_nhwc (N, C, W*H, h_X, r_sum, h_sumsq);
      bool ok = check(C, h_sum, r_sum);

      printf("Average time of channel sum (nhwc): %f (ms)\n", (time * 1e-6f) / repeat);
      printf("Verification %s for channel sum (nhwc)\n", ok ? "PASS" : "FAIL");
      if (!ok) exit(1);

      ComputeChannelSumNCHW (N, C, W*H, d_X, d_sum, d_sumsq, time, repeat);

      cudaMemcpy(h_sum, d_sum, output_size_bytes, cudaMemcpyDeviceToHost);
      ref_nchw (N, C, W*H, h_X, r_sum, h_sumsq);
      ok = check(C, h_sum, r_sum);
      
      printf("Average time of channel sum (nchw): %f (ms)\n", (time * 1e-6f) / repeat);
      printf("Verification %s for channel sum (nchw)\n", ok ? "PASS" : "FAIL");
      if (!ok) exit(1);

      cudaFree(d_X);
      cudaFree(d_sum);
      cudaFree(d_sumsq);

      free(h_X);
      free(h_sum);
      free(r_sum);
      free(h_sumsq);
    }
  }
  
  end: return 0;
}
