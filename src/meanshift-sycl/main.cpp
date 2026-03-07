#include <stdio.h>
#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>
#include "utils.h"
#include "constants.h"
#include <cstdlib>

namespace mean_shift::gpu {
  void mean_shift(sycl::nd_item<1> &item, const float *data, float *data_next) {
    size_t tid = item.get_global_id(0);
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
          float weight = sycl::exp(-sq_dist / DBL_SIGMA_SQ);
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

  void mean_shift_tiling(sycl::nd_item<1> &item,
                         float *__restrict local_data,
                         float *__restrict valid_data,
                         const float* data,
                               float*__restrict data_next) {

    // Shared memory allocation
    // A few convenient variables
    int tid = item.get_global_id(0);
    int lid = item.get_local_id(0);
    int row = tid * D;
    int local_row = lid * D;
    float new_position[D] = {0.f};
    float tot_weight = 0.f;
    // Load data in shared memory
    for (int t = 0; t < BLOCKS; ++t) {
      int tid_in_tile = t * TILE_WIDTH + lid;
      if (tid_in_tile < N) {
        int row_in_tile = tid_in_tile * D;
        for (int j = 0; j < D; ++j) {
          local_data[local_row + j] = data[row_in_tile + j];
        }
        valid_data[lid] = 1;
      }
      else {
        for (int j = 0; j < D; ++j) {
          local_data[local_row + j] = 0;
        }
        valid_data[lid] = 0;
      }
      item.barrier(sycl::access::fence_space::local_space);
      for (int i = 0; i < TILE_WIDTH; ++i) {
        int local_row_tile = i * D;
        float valid_radius = RADIUS * valid_data[i];
        float sq_dist = 0.;
        for (int j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - local_data[local_row_tile + j]) *
                     (data[row + j] - local_data[local_row_tile + j]);
        }
        if (sq_dist <= valid_radius) {
          float weight = sycl::exp(-sq_dist / DBL_SIGMA_SQ);
          for (int j = 0; j < D; ++j) {
            new_position[j] += (weight * local_data[local_row_tile + j]);
          }
          tot_weight += (weight * valid_data[i]);
        }
      }
      item.barrier(sycl::access::fence_space::local_space);
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
  std::array<float, N * D> result {};

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Allocate GPU memory
  size_t data_bytes = N * D * sizeof(float);
  float *d_data = (float*) sycl::malloc_device (data_bytes, q);
  float *d_data_next = (float*) sycl::malloc_device (data_bytes, q);;

  // Copy to GPU memory
  q.memcpy(d_data, data.data(), data_bytes).wait();

  sycl::range<1> gws (BLOCKS * THREADS);
  sycl::range<1> lws (THREADS);

  // Run mean shift clustering and time the execution
  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class base>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        mean_shift::gpu::mean_shift(item, d_data, d_data_next);
      });
    }).wait();
    mean_shift::gpu::utils::swap(d_data, d_data_next);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (base) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  // Verify these centroids are sufficiently close to real ones
  q.memcpy(result.data(), d_data, data_bytes).wait();
  auto centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  bool are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  // Reset device data
  q.memcpy(d_data, data.data(), data_bytes).wait();

  start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < mean_shift::gpu::NUM_ITER; ++i) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> local_data (sycl::range<1>(TILE_WIDTH * D), cgh);
      sycl::local_accessor<float, 1> valid_data (sycl::range<1>(TILE_WIDTH), cgh);
      cgh.parallel_for<class opt>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        mean_shift::gpu::mean_shift_tiling(item, local_data.get_multi_ptr<sycl::access::decorated::no>().get(),
                                           valid_data.get_multi_ptr<sycl::access::decorated::no>().get(),
                                           d_data, d_data_next);
      });
    }).wait();
    mean_shift::gpu::utils::swap(d_data, d_data_next);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "\nAverage execution time of mean-shift (opt) "
            << (time * 1e-6f) / mean_shift::gpu::NUM_ITER << " ms\n" << std::endl;

  // Verify these centroids are sufficiently close to real ones
  q.memcpy(result.data(), d_data, data_bytes).wait();
  centroids = mean_shift::gpu::utils::reduce_to_centroids<N, D>(result, mean_shift::gpu::MIN_DISTANCE);
  are_close = mean_shift::gpu::utils::are_close_to_real<M, D>(centroids, real, DIST_TO_REAL);
  if (centroids.size() == M && are_close)
     std::cout << "PASS\n";
  else
     std::cout << "FAIL\n";

  sycl::free(d_data, q);
  sycl::free(d_data_next, q);
  if (centroids.size() != M && are_close) return 1;
  if (centroids.size() != M && are_close) return 1;
  return 0;
}
