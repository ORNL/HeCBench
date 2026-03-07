#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

#define max(a, b) ((a) > (b) ? (a) : (b))

#pragma omp declare target
template<typename T, typename C>
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
#pragma omp end declare target

template <typename scalar_t, typename accscalar_t, typename outscalar_t>
void welford_kernel(
      const scalar_t* __restrict__ input,
      outscalar_t* __restrict__ out_mean,
      outscalar_t* __restrict__ out_var_biased,
      const int bs,
      const int fs,
      const int ss)
{
  #pragma omp target teams num_teams(fs) thread_limit(512)
  {
    int count_sm[512];
    accscalar_t mean_sm[512];
    accscalar_t var_sm[512];
    #pragma omp parallel
    {
      const int blockDim_x = 32;
      const int blockDim_y = 16;
      const int blockIdx_x = omp_get_team_num();
      int tid = omp_get_thread_num();
      int threadIdx_x = tid % blockDim_x;
      int threadIdx_y = tid / blockDim_x;
      int count = 0;
      accscalar_t x_mean = accscalar_t(0);
      accscalar_t m_2_n = accscalar_t(0);

      for (int batch_id = threadIdx_y; batch_id < bs; batch_id += blockDim_y) {
        int input_base = blockIdx_x * ss + batch_id * ss * fs;
        for (int offset = threadIdx_x; offset < ss ; offset += blockDim_x) {
          count++;
          auto x_n = static_cast<accscalar_t>(input[offset+input_base]);
          auto d = x_n - x_mean;
          x_mean += d / count;
          m_2_n += d * (x_n - x_mean);
        }
      }
      mean_sm[tid] = x_mean;
      var_sm[tid] = m_2_n;
      count_sm[tid] = count;
      #pragma omp barrier

      if (tid == 0) {
        auto num = count_sm[0];
        auto mean = mean_sm[0];
        auto m2n = var_sm[0];
        for (int i = 1; i < 512; i++) {
          welford_merge_element(num, mean, m2n, count_sm[i], mean_sm[i], var_sm[i]);
        }
        out_mean[blockIdx_x] = static_cast<outscalar_t>(mean);
        out_var_biased[blockIdx_x] = static_cast<outscalar_t>(m2n / num);
      }
    }
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
  
  #pragma omp target data map(to: input[0:input_size]) \
                          map(from: mean[0:feature_size]) \
                          map(from: var[0:feature_size])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      welford_kernel<float, float, float>(
      input, mean, var, batch_size, feature_size, spatial_size);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);
  }

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

  free(input);
  free(mean);
  free(var);
  free(r_mean);
  free(r_var);
  if (!ok) return 1;
  return 0;
}
