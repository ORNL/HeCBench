#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include "reference.h"

// calculate LB_Keogh
__global__
void lb_keogh(const float *__restrict__ subject,
              const float *__restrict__ avgs,
              const float *__restrict__ stds, 
                    float *__restrict__ lb_keogh,
              const float *__restrict__ lower_bound,
              const float *__restrict__ upper_bound,
              const int M,
              const int N) 
{
  // shared memory
  extern __shared__ float cache[];

  int lid = threadIdx.x;
  int blockSize = blockDim.x * blockIdx.x;
  int idx = blockSize + lid;

  for (int k = lid; k < blockDim.x + M; k += blockDim.x)
    if (blockSize + k < N) cache[k] = subject[blockSize + k];

  __syncthreads();

  if (idx < N-M+1) {

    // obtain statistics
    float residues = 0;
    float avg = avgs[idx];
    float std = stds[idx];

    for (int i = 0; i < M; ++i) {
      // differences to envelopes
      float value = (cache[lid+i] - avg) / std;
      float lower = value - lower_bound[i];
      float upper = value - upper_bound[i];

      // Euclidean or Manhattan distance?
      residues += upper*upper*(upper > 0) + lower*lower*(lower < 0);
    }

    lb_keogh[idx] = residues;
  }
}

int main(int argc, char* argv[]) {

  if (argc != 4) {
    printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return 1;
  }

  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Query length = %d\n", M);
  printf("Subject length = %d\n", N);

  // host side memory
  float *subject = (float*) malloc (sizeof(float)*N);
  float *lower = (float*) malloc (sizeof(float)*N);
  float *upper = (float*) malloc (sizeof(float)*N);
  float *lb = (float*) malloc (sizeof(float)*(N-M+1));
  float *lb_h = (float*) malloc (sizeof(float)*(N-M+1));
  float *avgs = (float*) malloc (sizeof(float)*(N-M+1));
  float *stds = (float*) malloc (sizeof(float)*(N-M+1));

  srand(123);
  for (int i = 0; i < N; ++i) subject[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) avgs[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < N-M+1; ++i) stds[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) upper[i] = (float)rand() / (float)RAND_MAX;
  for (int i = 0; i < M; ++i) lower[i] = (float)rand() / (float)RAND_MAX;

  float *d_subject = NULL, *d_avgs = NULL, *d_stds = NULL, 
        *d_lb = NULL, *d_lower = NULL, *d_upper = NULL;

  hipMalloc(&d_subject, sizeof(float)*N);
  hipMalloc(&d_avgs, sizeof(float)*(N-M+1));
  hipMalloc(&d_stds, sizeof(float)*(N-M+1));
  hipMalloc(&d_lb, sizeof(float)*(N-M+1));
  hipMalloc(&d_lower, sizeof(float)*M);
  hipMalloc(&d_upper, sizeof(float)*M);

  hipMemcpy(d_subject, subject, sizeof(float)*N, hipMemcpyHostToDevice);
  hipMemcpy(d_avgs, avgs, sizeof(float)*(N-M+1), hipMemcpyHostToDevice);
  hipMemcpy(d_stds, stds, sizeof(float)*(N-M+1), hipMemcpyHostToDevice);
  hipMemcpy(d_lower, lower, sizeof(float)*M, hipMemcpyHostToDevice);
  hipMemcpy(d_upper, upper, sizeof(float)*M, hipMemcpyHostToDevice);

  const int blocks = 256;
  const int grids = (N-M+1 + blocks - 1) / blocks;
  int smem_size = (M+blocks)*sizeof(float);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    hipLaunchKernelGGL(lb_keogh, grids, blocks, smem_size, 0, d_subject, d_avgs, d_stds, d_lb, d_lower, d_upper, M, N);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  hipMemcpy(lb, d_lb, sizeof(float)*(N-M+1), hipMemcpyDeviceToHost);

  // verify
  reference(subject, avgs, stds, lb_h, lower, upper, M, N);
  bool ok = true;
  for (int i = 0; i < N-M+1; i++) {
    if (fabsf(lb[i] - lb_h[i]) > 1e-2f) {
      printf("%d %f %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  hipFree(d_lb);
  hipFree(d_avgs);
  hipFree(d_stds);
  hipFree(d_subject);
  hipFree(d_lower);
  hipFree(d_upper);
  free(lb);
  free(lb_h);
  free(avgs);
  free(stds);
  free(subject);
  free(lower);
  free(upper);
  if (!ok) return 1;
  return 0;
}
