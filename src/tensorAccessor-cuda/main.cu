#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "tensorAccessor.h"

// A demo of packed tensor accessors in Pytorch
__global__
void tensor_packed_accessor_kernel (
    PackedTensorAccessor64<float, 1, RestrictPtrTraits> r,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> m,
    PackedTensorAccessor64<float, 1, RestrictPtrTraits> v)
{
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < r.size(0)) {
    float val = 0.0f;
    for (int64_t j = 0; j < m.size(1); j++) {
      val += m[i][j] * v[j];
    }
    r[i] = val;
  }
}

__global__
void raw_accessor_kernel (
    const int64_t nrow,
    const int64_t ncol,
          float *__restrict__ r,
    const float *__restrict__ m,
    const float *__restrict__ v)
{
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nrow) {
    float val = 0.0f;
    for (int64_t j = 0; j < ncol; j++) {
      val += m[i * ncol + j] * v[j];
    }
    r[i] = val;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }

  const int64_t nrow = atol(argv[1]);
  const int64_t ncol = atol(argv[2]);
  const int repeat = atoi(argv[3]);

  // tensor sizes and strides
  const int64_t sizes[2] = {nrow, ncol};
  const int64_t strides[2] = {ncol, 1};

  int64_t numel = 1;
  for (int i = 0; i < 2; i++) numel *= sizes[i];

  // matrix vector multiply
  int64_t m_bytes = numel * sizeof(float);
  int64_t v_bytes = ncol * sizeof(float);
  int64_t r_bytes = nrow * sizeof(float);

  float *m, *v, *r, *r_ref;
  m = (float*) malloc (m_bytes);
  v = (float*) malloc (v_bytes);
  r = (float*) malloc (r_bytes);
  r_ref = (float*) malloc (r_bytes);

  srand(123);
  for (int64_t i = 0; i < numel; i++) {
    m[i] = rand() / (float)RAND_MAX;
  }

  for (int64_t i = 0; i < ncol; i++) {
    v[i] = rand() / (float)RAND_MAX;
  }

  for (int64_t i = 0; i < nrow; i++) {
    float val = 0.f;
    for (int64_t j = 0; j < ncol; j++) {
      val += m[i * ncol + j] * v[j];
    }
    r_ref[i] = val;
  }

  float *d_m, *d_v, *d_r;
  cudaMalloc((void**)&d_m, m_bytes);
  cudaMemcpy(d_m, m, m_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_v, v_bytes);
  cudaMemcpy(d_v, v, v_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_r, r_bytes);

  PackedTensorAccessor64<float, 2, RestrictPtrTraits> m_acc (d_m, sizes, strides);
  PackedTensorAccessor64<float, 1, RestrictPtrTraits> v_acc (d_v, &ncol, strides+1);
  PackedTensorAccessor64<float, 1, RestrictPtrTraits> r_acc (d_r, &nrow, strides+1);

  dim3 grid ((nrow + 255) / 256);
  dim3 block (256);

  printf("Warmup..\n");
  for (int i = 0; i < repeat; i++) {
    tensor_packed_accessor_kernel<<<grid, block>>>(r_acc, m_acc, v_acc);
    raw_accessor_kernel<<<grid, block>>>(nrow, ncol, d_r, d_m, d_v);
  }

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    raw_accessor_kernel<<<grid, block>>>(nrow, ncol, d_r, d_m, d_v);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of raw_accessor_kernel: %f (us)\n", 
          time * 1e-3f / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    tensor_packed_accessor_kernel<<<grid, block>>>(r_acc, m_acc, v_acc);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of tensor_packed_accessor_kernel: %f (us)\n", 
          time * 1e-3f / repeat);

  cudaMemcpy(r, d_r, r_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_m);
  cudaFree(d_v);
  cudaFree(d_r);

  // verify (may fail due to floating-point rounding)
  bool ok = true;
  for (int64_t i = 0; i < nrow; i++) {
    if (fabsf(r[i] - r_ref[i]) > 1e-3f) {
      printf("%f %f\n", r[i], r_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(m);
  free(v);
  free(r);
  free(r_ref);
  
  if (!ok) return 1;
  return 0;
}
