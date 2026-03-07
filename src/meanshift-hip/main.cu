#include <stdio.h>
#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>
#include "utils.h"
#include "constants.h"
#include <cstdlib>

namespace mean_shift::gpu {
  __global__ void mean_shift(const float *data, float *data_next) {
    size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N) {
      size_t row = tid * D;
      float new_position[D] = {0.f};
      float tot_weight = 0.f;
      for (size_t i = 0; i < N; ++i) {
        size_t row_n = i * D;
        float sq_dist = 0.f;
        for (size_t j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - data[row_n + j]) * (data[row + j] - data[row_n + j]);
        }
        if (sq_dist <= RADIUS) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          for (size_t j = 0; j < D; ++j) {
            new_position[j] += weight * data[row_n + j];
          }
          tot_weight += weight;
        }
      }
      for (size_t j = 0; j < D; ++j) {
        data_next[row + j] = new_position[j] / tot_weight;
      }
    }
  }

  __global__ void mean_shift_tiling(const float* data, float* data_next) {

    // Shared memory allocation
    __shared__ float local_data[TILE_WIDTH * D];
    __shared__ float valid_data[TILE_WIDTH];
    // A few convenient variables
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int row = tid * D;
    int local_row = threadIdx.x * D;
    float new_position[D] = {0.f};
    float tot_weight = 0.f;
    // Load data in shared memory
    for (int t = 0; t < BLOCKS; ++t) {
      int tid_in_tile = t * TILE_WIDTH + threadIdx.x;
      if (tid_in_tile < N) {
        int row_in_tile = tid_in_tile * D;
        for (int j = 0; j < D; ++j) {
          local_data[local_row + j] = data[row_in_tile + j];
        }
        valid_data[threadIdx.x] = 1;
      }
      else {
        for (int j = 0; j < D; ++j) {
          local_data[local_row + j] = 0;
        }
        valid_data[threadIdx.x] = 0;
      }
      __syncthreads();
      for (int i = 0; i < TILE_WIDTH; ++i) {
        int local_row_tile = i * D;
        float valid_radius = RADIUS * valid_data[i];
        float sq_dist = 0.;
        for (int j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - local_data[local_row_tile + j]) *
                     (data[row + j] - local_data[local_row_tile + j]);
        }
        if (sq_dist <= valid_radius) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          for (int j = 0; j < D; ++j) {
            new_position[j] += (weight * local_data[local_row_tile + j]);
          }
          tot_weight += (weight * valid_data[i]);
        }
      }
      __syncthreads();
    }
    if (tid < N) {
      for (int j = 0; j < D; ++j) {
        data_next[row + j] = new_position[j] / tot_weight;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path to data> <path to centroids>" << std::endl;
    return 1;
  }
  const auto path_to_data = argv[1];
  const auto path_to_centroids = argv[2];

  constexpr auto N = mean_shift::gpu::N;
  constexpr auto D = mean_shift::gpu::D;
  constexpr auto M = mean_shift::gpu::M;
  constexpr auto THREADS = mean_shift::gpu::THREADS;
  constexpr auto BLOCKS = mean_shift::gpu::BLOCKS;
  constexpr auto TILE_WIDTH = mean_shift::gpu::TILE_WIDTH;
  constexpr auto DIST_TO_REAL = mean_shift::gpu::DIST_TO_REAL;

  mean_shift::gpu::utils::print_info(path_to_data, N, D, BLOCKS, THREADS, TILE_WIDTH);

  // Load data
  const std::array<float, M * D> real = mean_shift::gpu::utils::load_csv<M, D>(path_to_centroids, ',');
  std::array<float, N * D> data = mean_shift::gpu::utils::load_csv<N, D>(path_to_data, ',');
  std::array<float, N * D> result;

  // Allocate GPU memory
  float *d_data;
  float *d_data_next;
  size_t data_bytes = N * D * sizeof(float);
  hipMalloc((void**)&d_data, data_bytes);
  hipMalloc((void**)&d_data_next, data_bytes);

  // Copy to GPU memory
  hipMemcpy(d_data, data.data(), data_bytes, hipMemcpyHostToDevice);

  // Run mean shift clustering and time the execution
  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    hipLaunchKernelGGL(mean_shift::gpu::mean_shift, BLOCKS, THREADS, 0, 0, d_data, d_data_next);
    hipDeviceSynchronize();
    mean_shift::gpu::utils::swap(d_data, d_data_next);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (base) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  // Verify these centroids are sufficiently close to real ones
  hipMemcpy(result.data(), d_data, data_bytes, hipMemcpyDeviceToHost);
  auto centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  bool are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";
  if (centroids.size() != M && are_close) exit(1);

  // Reset device data
  hipMemcpy(d_data, data.data(), data_bytes, hipMemcpyHostToDevice);

  start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    hipLaunchKernelGGL(mean_shift::gpu::mean_shift_tiling, BLOCKS, THREADS, 0, 0, d_data, d_data_next);
    hipDeviceSynchronize();
    mean_shift::gpu::utils::swap(d_data, d_data_next);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (opt) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  // Verify these centroids are sufficiently close to real ones
  hipMemcpy(result.data(), d_data, data_bytes, hipMemcpyDeviceToHost);
  centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";
  if (centroids.size() != M && are_close) exit(1);


  hipFree(d_data);
  hipFree(d_data_next);
  return 0;
}
