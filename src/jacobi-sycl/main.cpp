// Copyright (c) 2021, NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <limits>
#include <utility>
#include <sycl/sycl.hpp>
#include <cstdlib>

// A multiple of thread block size
#define N 2048

#define IDX(i, j) ((i) + (j) * N)

void initialize_data (float* f) {
  // Set up simple sinusoidal boundary conditions
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {

      if (i == 0 || i == N-1) {
        f[IDX(i,j)] = sinf(j * 2 * M_PI / (N - 1));
      }
      else if (j == 0 || j == N-1) {
        f[IDX(i,j)] = sinf(i * 2 * M_PI / (N - 1));
      }
      else {
        f[IDX(i,j)] = 0.0f;
      }

    }
  }
}

void jacobi_step (float*__restrict__ f, 
                  const float*__restrict__ f_old, 
                  float*__restrict__ error,
                  const sycl::nd_item<2> &item,
                  float f_old_tile[18][18],
                  float *reduction_array) {

  int i = item.get_global_id(1);
  int j = item.get_global_id(0);
  int tx = item.get_local_id(1);
  int ty = item.get_local_id(0);

  // First read in the "interior" data, one value per thread
  // Note the offset by 1, to reserve space for the "left"/"bottom" halo

  f_old_tile[ty + 1][tx + 1] = f_old[IDX(i, j)];

  // Now read in the halo data; we'll pick the "closest" thread
  // to each element. When we do this, make sure we don't fall
  // off the end of the global memory array. Note that this
  // code does not fill the corners, as they are not used in
  // this stencil.

  if (tx == 0 && i >= 1) {
    f_old_tile[ty + 1][tx + 0] = f_old[IDX(i - 1, j)];
  }
  if (tx == 15 && i <= N - 2) {
    f_old_tile[ty + 1][tx + 2] = f_old[IDX(i + 1, j)];
  }
  if (ty == 0 && j >= 1) {
    f_old_tile[ty + 0][tx + 1] = f_old[IDX(i, j - 1)];
  }
  if (ty == 15 && j <= N - 2) {
    f_old_tile[ty + 2][tx + 1] = f_old[IDX(i, j + 1)];
  }

  // Synchronize all threads
  item.barrier(sycl::access::fence_space::local_space);

  float err = 0.0f;

  if (j >= 1 && j <= N-2) {
    if (i >= 1 && i <= N-2) {
      // Perform the read from shared memory
      f[IDX(i, j)] = 0.25f * (f_old_tile[ty + 1][tx + 2] +
                              f_old_tile[ty + 1][tx + 0] +
                              f_old_tile[ty + 2][tx + 1] +
                              f_old_tile[ty + 0][tx + 1]);
      float df = f[IDX(i, j)] - f_old_tile[ty + 1][tx + 1];
      err = df * df;
    }
  }

  // Sum over threads in the warp
  // For simplicity, we do this outside the above conditional
  // so that all threads participate
  auto sg = item.get_sub_group();
  for (int offset = 8; offset > 0; offset /= 2) {
    err += sycl::shift_group_left(sg, err, offset);
  }

  // If we're thread 0 in the warp, update our value to shared memory
  // Note that we're assuming exactly a 16x16 block and that the warp ID
  // is equivalent to threadIdx.y. For the general case, we would have to
  // write more careful code.

  if (tx == 0) {
    reduction_array[ty] = err;
  }

  // Synchronize the block before reading any values from smem
  item.barrier(sycl::access::fence_space::local_space);

  // Using the first warp in the block, reduce over the partial sums
  // in the shared memory array.
  if (ty == 0) {
    err = reduction_array[tx];
    for (int offset = 8; offset > 0; offset /= 2) {
      err += sycl::shift_group_left(sg, err, offset);
    }
    if (tx == 0) {
      //dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(error, err);
      auto ao = sycl::atomic_ref<float,
               sycl::memory_order::relaxed,
               sycl::memory_scope::device,
               sycl::access::address_space::global_space> (error[0]);
      ao.fetch_add(err);
    }
  }
}

int main() {
  // Begin wall timing
  auto start_time = std::chrono::steady_clock::now();

  // Reserve space for the scalar field and the "old" copy of the data
  float* f = (float*) aligned_alloc(64, N * N * sizeof(float));
  float* f_old = (float*) aligned_alloc(64, N * N * sizeof(float));

  // Initialize data (we'll do this on both f and f_old, so that we don't
  // have to worry about the boundary points later)
  initialize_data(f);
  initialize_data(f_old);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_f = sycl::malloc_device<float>(N * N, q);
  q.memcpy(d_f, f, N * N * sizeof(float));

  float *d_f_old = sycl::malloc_device<float>(N * N, q);
  q.memcpy(d_f_old, f_old, N * N * sizeof(float));

  float *d_error = sycl::malloc_device<float>(1, q);

  // Initialize error to a large number
  float error = std::numeric_limits<float>::max();
  const float tolerance = 1.e-5f;

  // Iterate until we're converged (but set a cap on the maximum number of
  // iterations to avoid any possible hangs)
  const int max_iters = 10000;
  int num_iters = 0;

  sycl::range<2> gws (N, N);
  sycl::range<2> lws (16, 16);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  while (error > tolerance && num_iters < max_iters) {
    // Initialize error to zero (we'll add to it the following step)
    q.memset(d_error, 0, 4);

    // Perform a Jacobi relaxation step
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float[18][18], 0> f_old_tile (cgh);
      sycl::local_accessor<float, 1> reduction_array(sycl::range<1>(16), cgh);
      cgh.parallel_for(
        sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> item) {
          jacobi_step(d_f, d_f_old, d_error, item, f_old_tile,
                      reduction_array.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });

    // Swap the old data and the new data
    std::swap(d_f, d_f_old);

    q.memcpy(&error, d_error, sizeof(float)).wait();

    // Normalize the L2-norm of the error by the number of data points
    // and then take the square root
    error = sqrtf(error / (N * N));

    // Periodically print out the current error
    if (num_iters % 1000 == 0) {
      std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
    }

    // Increment the iteration count
    ++num_iters;
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";

  // If we took fewer than max_iters steps and the error is below the tolerance,
  // we succeeded. Otherwise, we failed.

  if (error <= tolerance && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
  }
  else {
    std::cout << "FAIL" << std::endl;
    exit(1);
    return -1;
  }

  // CLean up memory allocations
  sycl::free(d_f, q);
  sycl::free(d_f_old, q);
  sycl::free(d_error, q);
  free(f);
  free(f_old);

  // End wall timing
  auto end_time = std::chrono::steady_clock::now();
  auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
  double duration = total_time * 1e-9;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
