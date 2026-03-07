#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <chrono>
#include <math.h>
#include <sycl/sycl.hpp>
#include "reference.h"

template<typename T, typename C>
inline
void welford_merge_element(C& count,
                           T& mean,
                           T& m2n,
                           const C& num_new,
                           const T& mean_new,
                           const T& m2n_new) {
  T factor = T(1.0) / sycl::max(1, (count + num_new));
  T delta0 = mean - mean_new;
  mean = (mean_new * num_new + mean * count) * factor;
  m2n += m2n_new + delta0 * delta0 * num_new * count * factor;
  count += num_new;
}

template<typename T>
inline
void warp_reduce_mean_m2n(sycl::sub_group &sg, T &mean, T &m2n, int &num)
{
  #pragma unroll
  for(int i = sg.get_max_local_range()[0] / 2; i > 0; i >>= 1) {
    auto num_new = sycl::shift_group_left(sg, num, i);
    auto mean_new = sycl::shift_group_left(sg, mean, i);
    auto m2n_new = sycl::shift_group_left(sg, m2n, i);
    welford_merge_element(num, mean, m2n, num_new, mean_new, m2n_new);
  }
}

template <typename T>
void welford_reduce_mean_m2n(
      sycl::nd_item<2> &item,
      T* __restrict x,
      int* __restrict count,
      T &mean,
      T &m2n,
      int &num,
      int block_size,
      int thread_id)
{
  sycl::sub_group warp = item.get_sub_group();
  int lane = warp.get_local_linear_id();
  int wid = warp.get_group_linear_id();

  if (block_size > warp.get_max_local_range()[0]) {
    warp_reduce_mean_m2n(warp, mean, m2n, num);
    if (lane == 0) {
      x[wid*2] = mean;
      x[wid*2+1] = m2n;
      count[wid] = num;
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (wid == 0) {
      mean = (thread_id < warp.get_group_linear_range()) ? x[lane*2] : T(0);
      m2n = (thread_id < warp.get_group_linear_range()) ? x[lane*2+1] : T(0);
      num = (thread_id < warp.get_group_linear_range()) ? count[lane] : int(0);
    }
  }

  if (wid==0) warp_reduce_mean_m2n(warp, mean, m2n, num);
}

template <typename scalar_t, typename accscalar_t, typename outscalar_t>
void welford_kernel(
      sycl::nd_item<2> &item,
      const scalar_t* __restrict input,
      outscalar_t* __restrict out_mean,
      outscalar_t* __restrict out_var_biased,
      int *s_mem,
      const int bs,
      const int fs,
      const int ss)
{
  int blockDim_x = item.get_local_range(1);
  int blockDim_y = item.get_local_range(0);
  int threadIdx_x = item.get_local_id(1);
  int threadIdx_y = item.get_local_id(0);
  int blockIdx_x = item.get_group(1);

  int block_size = blockDim_x * blockDim_y;
  int count = 0;
  accscalar_t x_mean = accscalar_t(0);
  accscalar_t m_2_n = accscalar_t(0);

  int thread_id = threadIdx_y*blockDim_x + threadIdx_x;

  for (int batch_id = threadIdx_y; batch_id < bs; batch_id += blockDim_y) {
    int input_base = blockIdx_x*ss + batch_id*ss*fs;
    for (int offset = threadIdx_x; offset < ss ; offset += blockDim_x) {
      count++;
      auto x_n = static_cast<accscalar_t>(input[offset+input_base]);
      // sequential welford
      auto d = x_n - x_mean;
      x_mean += d / count;
      m_2_n += d * (x_n - x_mean);
    }
  }

  accscalar_t* s_mem_ac = (accscalar_t*) (s_mem + 
    item.get_sub_group().get_max_local_range()[0] / 2);

  welford_reduce_mean_m2n<accscalar_t>(
    item, s_mem_ac, s_mem, x_mean, m_2_n, count, block_size, thread_id);

  if (thread_id == 0) {
    out_mean[blockIdx_x] = static_cast<outscalar_t>(x_mean);
    out_var_biased[blockIdx_x] = static_cast<outscalar_t>(m_2_n/count);
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

  sycl::range<2> lws (block_y, block_x);
  sycl::range<2> gws (block_y, feature_size * block_x);

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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_input, *d_mean, *d_var;
  d_input = (float*) sycl::malloc_device(is_bytes, q);
  q.memcpy(d_input, input, is_bytes);

  d_mean = (float*) sycl::malloc_device(fs_bytes, q);
  d_var = (float*) sycl::malloc_device(fs_bytes, q);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<int, 1> sm (160, cgh);
      cgh.parallel_for<class welford>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item)
        //[[intel::reqd_sub_group_size(WARP_SIZE)]] 
      {
        welford_kernel<float, float, float>(
          item, d_input, d_mean, d_var, 
          sm.get_multi_ptr<sycl::access::decorated::no>().get(),
          batch_size, feature_size, spatial_size);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  q.memcpy(var, d_var, fs_bytes);
  q.memcpy(mean, d_mean, fs_bytes);
  q.wait();

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
  if (!ok) exit(1);

  sycl::free(d_input, q);
  sycl::free(d_mean, q);
  sycl::free(d_var, q);
  free(input);
  free(mean);
  free(var);
  free(r_mean);
  free(r_var);
  return 0;
}
