#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <vector>

#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template <typename T>
__device__ __forceinline__ bool is_aligned(T* p) {
  return ((uint64_t)p) % (ILP * sizeof(T)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset) {
  typedef typename std::aligned_storage<ILP * sizeof(T), ILP * alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template <typename x_t, typename y_t, typename out_t>
struct AxpbyFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int* noop_gmem, TensorListMetadata<3>& tl,
                                             float a, float b, int arg_to_check) {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t* x = (x_t*)tl.addresses[0][tensor_loc];
    x += chunk_idx * chunk_size;

    y_t* y = (y_t*)tl.addresses[1][tensor_loc];
    y += chunk_idx * chunk_size;

    out_t* out = (out_t*)tl.addresses[2][tensor_loc];
    out += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    bool finite = true;
    x_t r_x[ILP];
    y_t r_y[ILP];
    out_t r_out[ILP];

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x) && is_aligned(y) && is_aligned(out)) {
      for (int i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size; i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
        load_store(r_y, y, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          r_out[ii] = a * static_cast<float>(r_x[ii]) + b * static_cast<float>(r_y[ii]);
          if (arg_to_check == -1) finite = finite && (isfinite(r_x[ii]) && isfinite(r_y[ii]));
          if (arg_to_check == 0) finite = finite && isfinite(r_x[ii]);
          if (arg_to_check == 1) finite = finite && isfinite(r_y[ii]);
        }
        // store
        load_store(out, r_out, i_start, 0);
      }
    } else {
      // Non-divergent exit condition for __syncthreads, not necessary here
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          r_x[ii] = 0;
          r_y[ii] = 0;
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            r_x[ii] = x[i];
            r_y[ii] = y[i];
          }
        }
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          r_out[ii] = a * static_cast<float>(r_x[ii]) + b * static_cast<float>(r_y[ii]);
          if (arg_to_check == -1) finite = finite && (isfinite(r_x[ii]) && isfinite(r_y[ii]));
          if (arg_to_check == 0) finite = finite && isfinite(r_x[ii]);
          if (arg_to_check == 1) finite = finite && isfinite(r_y[ii]);
        }
        // see note in multi_tensor_scale_kernel.cu
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) out[i] = r_out[ii];
        }
      }
    }
    if (!finite) *noop_gmem = 1;  // Blindly fire off a write.  These will race but that's ok.
  }
};

template<typename scalar_t_0, typename scalar_t_1, typename scalar_t_2>
void multi_tensor_axpby(int chunk_size, Tensor<int> noop_flag,
                        std::vector<std::vector<Tensor<scalar_t_2>>> &tensor_lists,
                        float a, float b, int arg_to_check) {

  auto start = std::chrono::steady_clock::now();

  multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                        AxpbyFunctor<scalar_t_0, scalar_t_1, scalar_t_2>(), a, b, arg_to_check);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Chunk size %8d | Total execution time of multi_tensor_axpby: %f (us)\n", chunk_size, (time * 1e-3f));
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <number of tensors\n", argv[0]);
    return 1;
  }
  const int max_tensors = atoi(argv[1]);

  // small tensors
  std::vector<std::vector<Tensor<float>>> tensor_lists_ref (3);
  std::vector<std::vector<Tensor<float>>> tensor_lists (3);
  srand(123);
  for (int n = 0; n < max_tensors; n++) {
    int64_t length = rand() % (1024*1024) + 1024;
    for (int d = 0; d < 3; d++) {
      float *d_tensor;
      cudaMalloc(&d_tensor, sizeof(float) * length);
      float *tensor = (float*) malloc (sizeof(float) * length);
      if (d <= 1) {
        for (int64_t i = 0; i < length; i++)
          tensor[i] = rand() % length;
        cudaMemcpy(d_tensor, tensor, sizeof(float) * length, cudaMemcpyHostToDevice);
      }
      Tensor<float> t;
      t.data_ptr = d_tensor;
      t.numel = length;
      tensor_lists[d].push_back(t);

      Tensor<float> t_ref;
      t_ref.data_ptr = tensor;
      t_ref.numel = length;
      tensor_lists_ref[d].push_back(t_ref);
    }
  }

  Tensor<int> noop_flag;
  int *d_noop;
  cudaMalloc(&d_noop, sizeof(int));
  noop_flag.data_ptr = d_noop;
  noop_flag.numel = 1;

  const float a = 1.f;
  const float b = 1.f;
  int arg_to_check = -1;

  for (int chunk_size = 256; chunk_size <= 1024*1024; chunk_size = chunk_size * 2) {

    multi_tensor_axpby<float, float, float>(
      chunk_size, noop_flag, tensor_lists, a, b, arg_to_check);

    bool ok  = true;
    for (int n = 0; n < max_tensors; n++) {
      auto x = tensor_lists_ref[0][n];
      auto y = tensor_lists_ref[1][n];
      auto z = tensor_lists_ref[2][n];
      for (int i = 0; i < x.numel; i++) {
        z.data_ptr[i] = a * x.data_ptr[i] + b * y.data_ptr[i];
      }
      auto d_z = tensor_lists[2][n];
      std::vector<float> t (x.numel);
      cudaMemcpy(t.data(), d_z.data_ptr, x.numel * sizeof(float), cudaMemcpyDeviceToHost);
      for (int i = 0; i < x.numel; i++) {
        if (fabsf(t[i] - z.data_ptr[i]) > 1e-3f) {
          ok = false;
          break;
        }
      }
      if (!ok) break;
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  cudaFree(d_noop);
  for (int d = 0; d < 3; d++) {
    for (int t = 0; t < max_tensors; t++) {
       free(tensor_lists_ref[d][t].data_ptr);
       cudaFree(tensor_lists[d][t].data_ptr);
    }
  }

  if (!ok) return 1;
  return 0;
}
