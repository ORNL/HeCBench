/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <chrono>
#include <vector>
#include <hip/hip_runtime.h>

#include "reference.cpp"

// Find the best Gaussian bandwidth for each row in the dataset 
template <typename value_idx, typename value_t>
__global__
void sigmas_kernel(const value_t* __restrict__ distances,
                         value_t* __restrict__ P,
                   const float perplexity,
                   const float desired_entropy,
                   const int epochs,
                   const float tol,
                   const value_idx n,
                   const int k)
{
  // For every item in row
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  value_t beta_min = -INFINITY, beta_max = INFINITY;
  value_t beta = 1;
  const int ik = i * k;
  int step;

  for (step = 0; step < epochs; step++) {
    value_t sum_Pi = FLT_EPSILON;

    // Exponentiate to get Gaussian
    for (int j = 0; j < k; j++) {
      P[ik + j] = __expf(-distances[ik + j] * beta);
      sum_Pi += P[ik + j];
    }

    // Normalize
    value_t sum_disti_Pi = 0;
    const value_t div    = __fdividef(1.0f, sum_Pi);
    for (int j = 0; j < k; j++) {
      P[ik + j] *= div;
      sum_disti_Pi += distances[ik + j] * P[ik + j];
    }

    const value_t entropy      = __logf(sum_Pi) + beta * sum_disti_Pi;
    const value_t entropy_diff = entropy - desired_entropy;
    if (fabsf(entropy_diff) <= tol) {
      break;
    }

    // Bisection search
    if (entropy_diff > 0) {
      beta_min = beta;
      if (isinf(beta_max))
        beta *= 2.0f;
      else
        beta = (beta + beta_max) * 0.5f;
    } else {
      beta_max = beta;
      if (isinf(beta_min))
        beta *= 0.5f;
      else
        beta = (beta + beta_min) * 0.5f;
    }
  }
}

/****************************************/
template <typename value_idx, typename value_t>
void perplexity_search(const value_t* __restrict__ distances,
                       value_t* __restrict__ P,
                       const float perplexity,
                       const int epochs,
                       const float tol,
                       const value_idx n,
                       const int dim,
                       double &time)
{
  const float desired_entropy = logf(perplexity);
  dim3 grid ((n+255)/256);
  dim3 block (256);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  hipLaunchKernelGGL(sigmas_kernel, grid, block, 0, 0, distances, P, perplexity, desired_entropy, epochs, tol, n, dim);

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of points> <perplexity> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]); // points
  const int p = atoi(argv[2]); // perplexity
  const int repeat = atoi(argv[3]);

  const int n_nbrs = 4 * p;    // neighbors
  const int max_iter = 100;    // maximum number of iterations
  const float tol = 1e-8f;     // tolerance

  srand(123); 
  std::vector<float> data(n * n_nbrs);
  std::vector<float> h_data(n * n_nbrs);
  std::vector<float> distance(n * n_nbrs);
  for (int i = 0; i < n * n_nbrs; i++) {
    distance[i] = rand() / (float)RAND_MAX;
  }

  float *d_data;
  hipMalloc((void**)&d_data, sizeof(float)*n*n_nbrs);

  float *d_distance;
  hipMalloc((void**)&d_distance, sizeof(float)*n*n_nbrs);

  hipMemcpy(d_distance, distance.data(), sizeof(float)*n*n_nbrs, hipMemcpyHostToDevice); 

  double time = 0.0;

  for (int i = 0; i < repeat; i++)
    perplexity_search(d_distance, d_data, p, max_iter, tol, n, n_nbrs, time);

  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  hipMemcpy(data.data(), d_data, sizeof(float)*n*n_nbrs, hipMemcpyDeviceToHost); 

  // verify
  reference(distance.data(), h_data.data(), p, max_iter, tol, n, n_nbrs);

  bool ok = true;
  for (int i = 0; i < n*n_nbrs; i++) {
    if (fabsf(data[i] - h_data[i]) > 1e-3f) {
      printf("%d %f %f\n", i, data[i], h_data[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
  
  hipFree(d_distance);
  hipFree(d_data);
  return 0;
}
