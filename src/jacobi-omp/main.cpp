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

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <ctime>
#include <chrono>
#include <omp.h>
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

int main () {
  // Begin wall timing
  auto start_time = std::chrono::steady_clock::now();

  // Reserve space for the scalar field and the "old" copy of the data
  float* f = (float*) aligned_alloc(64, N * N * sizeof(float));
  float* f_old = (float*) aligned_alloc(64, N * N * sizeof(float));
  // Initialize error to a large number
  float error = {std::numeric_limits<float>::max()};
  const float tolerance = 1.e-5f;

  // Initialize data (we'll do this on both f and f_old, so that we don't
  // have to worry about the boundary points later)
  initialize_data(f);
  initialize_data(f_old);

  // Iterate until we're converged (but set a cap on the maximum number of
  // iterations to avoid any possible hangs)
  const int max_iters = 10000;
  int num_iters = 0;

#pragma omp target data map(to: f[0:N*N], f_old[0:N*N])
{
  auto start = std::chrono::steady_clock::now();

  while (error > tolerance && num_iters < max_iters) {
    // Initialize error to zero (we'll add to it the following step)
    // Perform a Jacobi relaxation step
    error = 0.f;
    
    #pragma omp target teams distribute parallel for collapse(2) \
     reduction(+:error) num_teams(N*N/256) thread_limit(256) map(tofrom: error)
    for (int i = 1; i <= N-2; i++) {
      for (int j = 1; j <= N-2; j++) {
        float t = 0.25f * (f_old[IDX(i-1,j)] +
                           f_old[IDX(i+1,j)] +
                           f_old[IDX(i,j-1)] +
                           f_old[IDX(i,j+1)]);
        float df = t - f_old[IDX(i, j)];
        f[IDX(i,j)] = t;
        error += df * df;
      }
    }
      
    // Swap the old data and the new data
    // We're doing this explicitly for pedagogical purposes, even though
    // in this specific application a std::swap would have been OK
    #pragma omp target teams distribute parallel for collapse(2) \
     thread_limit(256)
    for (int j = 0; j < N; j++) 
      for (int i = 0; i < N; i++) 
        if (j >= 1 && j <= N-2 && i >= 1 && i <= N-2)
          f_old[IDX(i,j)] = f[IDX(i,j)];

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

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";
}

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
  free(f);
  free(f_old);

  // End wall timing
  auto end_time = std::chrono::steady_clock::now();
  auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
  double duration = total_time * 1e-9;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
