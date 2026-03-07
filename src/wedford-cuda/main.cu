#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "reference.h"

template<typename T, typename C>
__device__ __forceinline__
void welford_merge_element(C& count,
                           T& mean,
                           T& m2n,
                           const C& num_new,
                           const T& mean_new,
                           const T& m2n_new) {
  T factor = T(1.0) / max(1, (count + num_new));
  T delta0 = mean - mean_new;
  mean = (mean_new * num_new + mean * count) * factor;
  m2n += m2n_new + delta0 * delta0 * num_new * count * factor;
  count += num_new;
}

template<typename T>
__device__ __forceinline__
void warp_reduce_mean_m2n(T &mean, T &m2n, int &num)
{
  #pragma unroll
  for(int i = warpSize/2; i > 0; i >>= 1) {
    auto num_new = __shfl_down_sync(0xffffffff, num, i);
    auto mean_new = __shfl_down_sync(0xffffffff, mean, i);
    auto m2n_new = __shfl_down_sync(0xffffffff, m2n, i);
    welford_merge_element(num, mean, m2n, num_new, mean_new, m2n_new);
  }
}

template <typename T>
__device__ void welford_reduce_mean_m2n(
      T* __restrict__ x,
      int* __restrict__ count,
      T &mean,
      T &m2n,
      int &num,
      int block_size,
      int thread_id)
{
  int lane = thread_id % warpSize;
  int wid = thread_id / warpSize;

  if (block_size > warpSize) {
    warp_reduce_mean_m2n(mean, m2n, num);
    if (lane == 0) {
      x[wid*2] = mean;
      x[wid*2+1] = m2n;
      count[wid] = num;
    }
    __syncthreads();

    if (wid == 0) {
      mean = (thread_id < block_size / warpSize)? x[lane*2] : T(0);
      m2n = (thread_id < block_size / warpSize)? x[lane*2+1] : T(0);
      num = (thread_id < block_size / warpSize)? count[lane] : int(0);
    }
  }

  if (wid==0) warp_reduce_mean_m2n(mean, m2n, num);
}

template <typename scalar_t, typename accscalar_t, typename outscalar_t>
__global__ void welford_kernel(
      const scalar_t* __restrict__ input,
      outscalar_t* __restrict__ out_mean,
      outscalar_t* __restrict__ out_var_biased,
      const int bs,
      const int fs,
      const int ss)
{
  int block_size = blockDim.x * blockDim.y;
  int count = 0;
  accscalar_t x_mean = accscalar_t(0);
  accscalar_t m_2_n = accscalar_t(0);

  int thread_id = threadIdx.y*blockDim.x + threadIdx.x;

  for (int batch_id = threadIdx.y; batch_id < bs; batch_id += blockDim.y) {
    int input_base = blockIdx.x*ss + batch_id*ss*fs;
    for (int offset = threadIdx.x; offset < ss ; offset += blockDim.x) {
      count++;
      auto x_n = static_cast<accscalar_t>(input[offset+input_base]);
      // sequential welford
      auto d = x_n - x_mean;
      x_mean += d / count;
      m_2_n += d * (x_n - x_mean);
    }
  }

  static __shared__ int s_mem[160];
  accscalar_t* s_mem_ac = (accscalar_t*) &s_mem[32];

  welford_reduce_mean_m2n<accscalar_t>(s_mem_ac, s_mem, x_mean, m_2_n, count, block_size, thread_id);

  if (thread_id == 0) {
    out_mean[blockIdx.x] = static_cast<outscalar_t>(x_mean);
    out_var_biased[blockIdx.x] = static_cast<outscalar_t>(m_2_n/count);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <batch_size> <spatial_size> <feature_size> <repeat>\n", argv[0]);
    return 1;
  }
    
  const int batch_size = atoi(argv[1]);
  const int spatial_size = atoi(argv[2]);
  const int feature_size = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int block_y = 16;
  const int block_x = 32;

  const dim3 block(block_x, block_y);
  const dim3 grid(feature_size, 1);

  int fs_bytes = feature_size * sizeof(float);
  size_t input_size = (size_t)batch_size * spatial_size * feature_size;
  size_t is_bytes = input_size * sizeof(float);

  float *input = (float*) malloc (is_bytes);

  srand(123);
  for (size_t i = 0; i < input_size; i++) {
    input[i] = rand() / (float)RAND_MAX;
  }

  float *mean = (float*) malloc (fs_bytes);
  float *var = (float*) malloc (fs_bytes);
  float *r_mean = (float*) malloc (fs_bytes);
  float *r_var = (float*) malloc (fs_bytes);
  
  float *d_input, *d_mean, *d_var;  
  cudaMalloc((void**)&d_input, is_bytes);
  cudaMemcpy(d_input, input, is_bytes, cudaMemcpyHostToDevice); 

  cudaMalloc((void**)&d_mean, fs_bytes);
  cudaMalloc((void**)&d_var, fs_bytes);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    welford_kernel<float, float, float><<<grid, block>>>(
      d_input, d_mean, d_var, batch_size, feature_size, spatial_size);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  cudaMemcpy(var, d_var, fs_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(mean, d_mean, fs_bytes, cudaMemcpyDeviceToHost);

  welford_reference<float, float, float>(
      input, r_mean, r_var, batch_size, feature_size, spatial_size);

  bool ok = true;
  for (int i = 0; i < feature_size; i++) {
    if (fabsf(var[i] - r_var[i]) > 1e-3f || fabsf(mean[i] - r_mean[i]) > 1e-3f) {
       printf("Error at index %d: %f %f %f %f\n", i, var[i], r_var[i], mean[i], r_mean[i]);
       ok = false;
       break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  cudaFree(d_input);
  cudaFree(d_mean);
  cudaFree(d_var);
  free(input);
  free(mean);
  free(var);
  free(r_mean);
  free(r_var);
  if (!ok) return 1;
  return 0;
}
