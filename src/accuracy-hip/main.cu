#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "reference.h"

#define GPU_NUM_THREADS 256

template <typename T>
__device__ void BlockReduce(T &input) {
  typedef hipcub::BlockReduce<T, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  input = BlockReduce(temp_storage).Sum(input);
}

__global__
void accuracy_kernel(
    const int N,
    const int D,
    const int top_k,
    const float* __restrict__ Xdata,
    const int* __restrict__ labelData,
    int* accuracy)
{
  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const int label = labelData[row];
    const float label_pred = Xdata[row * D + label];
    int ngt = 0;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      const float pred = Xdata[row * D + col];
      if (pred > label_pred || (pred == label_pred && col <= label)) {
        ++ngt;
      }
    }
    BlockReduce(ngt);
    if (ngt <= top_k) {
      ++count;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(accuracy, count);
  }
}

// modified from Claude Free
__global__
void accuracy_kernel2(
    const int N,
    const int D,
    const int top_k,
    const float* __restrict__ Xdata,
    const int*   __restrict__ labelData,
    int* accuracy)
{
  __shared__ float s_label_pred;
  __shared__ int s_label;

  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {

    if (threadIdx.x == 0) {
      s_label = labelData[row];
      s_label_pred = Xdata[row * D + s_label];
    }
    __syncthreads();

    const int   label      = s_label;
    const float label_pred = s_label_pred;

    int ngt = 0;
    const float* row_ptr = Xdata + row * D;
    int col = threadIdx.x;
    for (; col + 3 * blockDim.x < D; col += 4 * blockDim.x) {
      float p0 = row_ptr[col];
      float p1 = row_ptr[col +     blockDim.x];
      float p2 = row_ptr[col + 2 * blockDim.x];
      float p3 = row_ptr[col + 3 * blockDim.x];

      ngt += (p0 > label_pred || (p0 == label_pred && col                   <= label));
      ngt += (p1 > label_pred || (p1 == label_pred && col +     blockDim.x  <= label));
      ngt += (p2 > label_pred || (p2 == label_pred && col + 2 * blockDim.x  <= label));
      ngt += (p3 > label_pred || (p3 == label_pred && col + 3 * blockDim.x  <= label));
    }
    for (; col < D; col += blockDim.x) {
      float pred = row_ptr[col];
      ngt += (pred > label_pred || (pred == label_pred && col <= label));
    }

    BlockReduce(ngt);

    if (threadIdx.x == 0 && ngt <= top_k) {
      ++count;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && count > 0) {
    atomicAdd(accuracy, count);
  }
}


int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of rows> <number of columns> <top K> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ndims = atoi(argv[2]);
  const int top_k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int data_size = nrows * ndims;

  const int label_size_bytes = nrows * sizeof(int);
  const size_t data_size_bytes = data_size * sizeof(float);

  int *label = (int*) malloc (label_size_bytes);

  srand(123);
  for (int i = 0; i < nrows; i++)
    label[i] = rand() % ndims;

  float *data = (float*) malloc (data_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (0.f, 1.f);
  for (int i = 0; i < data_size; i++) {
    data[i] = distr(g);
  }

  int count_ref = reference(nrows, ndims, top_k, data, label);

  int *d_label;
  hipMalloc((void**)&d_label, label_size_bytes);
  hipMemcpy(d_label, label, label_size_bytes, hipMemcpyHostToDevice);

  float *d_data;
  hipMalloc((void**)&d_data, data_size_bytes);
  hipMemcpy(d_data, data, data_size_bytes, hipMemcpyHostToDevice);

  int *d_count;
  hipMalloc((void**)&d_count, sizeof(int));

  hipDeviceSynchronize();
  dim3 block (GPU_NUM_THREADS);

  for (int ngrid = nrows / 4; ngrid <= nrows; ngrid += nrows / 4) {

    dim3 grid (ngrid);
    printf("Grid size is %d\n", ngrid);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      hipMemset(d_count, 0, sizeof(int));
      accuracy_kernel<<<grid, block>>>(nrows, ndims, top_k, d_data, d_label, d_count);
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of accuracy kernel: %f (us)\n", (time * 1e-3f) / repeat);

    int count;
    hipMemcpy(&count, d_count, sizeof(int), hipMemcpyDeviceToHost);
    printf("%s\n", (count == count_ref) ? "PASS" : "FAIL");
    // printf("Accuracy = %f\n", (float)count / nrows);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      hipMemset(d_count, 0, sizeof(int));
      accuracy_kernel2<<<grid, block>>>(nrows, ndims, top_k, d_data, d_label, d_count);
    }

    hipDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of accuracy kernel2: %f (us)\n", (time * 1e-3f) / repeat);
    hipMemcpy(&count, d_count, sizeof(int), hipMemcpyDeviceToHost);
    printf("%s\n", (count == count_ref) ? "PASS" : "FAIL");
  }

  hipFree(d_label);
  hipFree(d_data);
  hipFree(d_count);

  free(label);
  free(data);

  return 0;
}
