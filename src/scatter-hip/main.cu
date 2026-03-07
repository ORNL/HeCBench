#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include "TensorInfo.h"
#include "reducer.h"

#define THREADS 256
#define BLOCKS(N) (N + THREADS - 1) / THREADS

#define CHECK_HIP(func)                                                       \
{                                                                             \
    hipError_t status = func;                                                 \
    if (status != hipSuccess) {                                               \
        printf("HIP API failed at line %d with error: %s (%d)\n",             \
               __LINE__, hipGetErrorString(status), status);                  \
    }                                                                         \
}

template <typename scalar_t, ReductionType REDUCE>
__global__ void
scatter_kernel(const scalar_t *src_data,
               const TensorInfo<int64_t, int64_t> index_info,
               scalar_t *out_data, int E, int K, int N, int numel) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int b = thread_idx / (E * K);
  int k = thread_idx % K;

  if (thread_idx < numel) {
    int offset = IndexToOffset<int64_t, int64_t, -1>::get(
        thread_idx, index_info);
    int64_t idx = index_info.data[offset];

    Reducer<scalar_t, REDUCE>::atomic_write(out_data + b * N * K + idx * K + k,
                                            src_data[thread_idx]);
  }
}

template<typename scalar_t, ReductionType REDUCE>
void scatter(int64_t num_elems, int repeat) {
  int64_t out_size = num_elems / 2;
  int64_t src_size_bytes = num_elems * sizeof(scalar_t);
  int64_t idx_size_bytes = num_elems * sizeof(int64_t);
  int64_t out_size_bytes = out_size * sizeof(scalar_t);

  int64_t *h_idx = (int64_t*) malloc (idx_size_bytes);
  scalar_t *h_src = (scalar_t*) malloc (src_size_bytes);
  scalar_t *h_out = (scalar_t*) malloc (out_size_bytes);
  scalar_t *h_ref = (scalar_t*) malloc (out_size_bytes);
  srand(123);
  for (int64_t i = 0; i < num_elems; i++) {
    h_idx[i] = rand() % out_size;
    h_src[i] = rand() % 2 ? 1 : -1;
  }

  scalar_t *d_src, *d_out;
  int64_t *d_idx;
  CHECK_HIP( hipMalloc(&d_src, src_size_bytes) )
  CHECK_HIP( hipMalloc(&d_out, out_size_bytes) )
  CHECK_HIP( hipMalloc(&d_idx, idx_size_bytes) )
  CHECK_HIP( hipMemcpy(d_idx, h_idx, idx_size_bytes, hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(d_src, h_src, src_size_bytes, hipMemcpyHostToDevice) )

  int64_t tensor_sizes[] = {num_elems};
  int64_t tensor_strides[] = {1};

  TensorInfo<int64_t, int64_t> index_info(d_idx, 1, tensor_sizes, tensor_strides);

  dim3 grids (BLOCKS(num_elems));
  dim3 blocks (THREADS);

  int B = 1;
  int E = num_elems;
  int K = num_elems / (B * E);
  int N = num_elems / 2;

  for (int64_t i = 0; i < out_size; i++) {
    h_ref[i] = h_out[i] = Reducer<scalar_t, REDUCE>::init();
  }
  
  int64_t time = 0;
  for (int i = 0; i < repeat; i++) {
    CHECK_HIP( hipMemcpy(d_out, h_out, out_size_bytes, hipMemcpyHostToDevice) )
    CHECK_HIP( hipDeviceSynchronize() )
    auto start = std::chrono::steady_clock::now();
    scatter_kernel<scalar_t, REDUCE> <<<grids, blocks>>>
        (d_src, index_info, d_out, E, K, N, num_elems);
    CHECK_HIP( hipDeviceSynchronize() )
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of kernel: %f (us)\n", (time * 1e-3f) / repeat);

  CHECK_HIP( hipMemcpy(h_out, d_out, out_size_bytes, hipMemcpyDeviceToHost) )

  for (int64_t i = 0; i < num_elems; i++) {
    Reducer<scalar_t, REDUCE>::update(h_ref[h_idx[i]], h_src[i]);
  }

  bool ok = true;
  for (int64_t i = 0; i < out_size; i++) {
    if ((double)h_out[i] - (double)h_ref[i] > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n\n", ok ? "PASS" : "FAIL");

  CHECK_HIP( hipFree(d_src) )
  CHECK_HIP( hipFree(d_idx) )
  CHECK_HIP( hipFree(d_out) )
  free(h_src);
  free(h_idx);
  free(h_out);
  free(h_ref);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int64_t num_elements = atol(argv[1]);
  const int repeat = atoi(argv[2]);
  printf("INT32 scatter (mul, div, sum, min, max)\n"); 
  scatter<int32_t, MUL>(num_elements, repeat);
  scatter<int32_t, DIV>(num_elements, repeat);
  scatter<int32_t, SUM>(num_elements, repeat);
  scatter<int32_t, MIN>(num_elements, repeat);
  scatter<int32_t, MAX>(num_elements, repeat);
  printf("INT64 scatter (mul, div, sum, min, max)\n"); 
  scatter<int64_t, MUL>(num_elements, repeat);
  scatter<int64_t, DIV>(num_elements, repeat);
  scatter<int64_t, SUM>(num_elements, repeat);
  scatter<int64_t, MIN>(num_elements, repeat);
  scatter<int64_t, MAX>(num_elements, repeat);
  printf("FP32 scatter (mul, div, sum, min, max)\n"); 
  scatter<float, MUL>(num_elements, repeat);
  scatter<float, DIV>(num_elements, repeat);
  scatter<float, SUM>(num_elements, repeat);
  scatter<float, MIN>(num_elements, repeat);
  scatter<float, MAX>(num_elements, repeat);
  printf("FP64 scatter (mul, div, sum, min, max)\n"); 
  scatter<double, MUL>(num_elements, repeat);
  scatter<double, DIV>(num_elements, repeat);
  scatter<double, SUM>(num_elements, repeat);
  scatter<double, MIN>(num_elements, repeat);
  scatter<double, MAX>(num_elements, repeat);
  if (!ok) return 1;
  return 0;
}
