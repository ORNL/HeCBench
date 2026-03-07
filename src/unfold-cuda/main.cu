#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda.h>

template <int n_threads, int n_elems_per_thread, typename func_t>
__global__
void unfold_backward_elementwise_kernel(int total_n_elems, func_t f) {
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  int idx = total_work_block * blockIdx.x + threadIdx.x;
  #pragma unroll
  for (int i = 0; i < n_elems_per_thread; ++i) {
    if (idx < total_n_elems) {
      f(idx);
      idx += n_threads;
    }
  }
}

template <int n_threads, int n_elems_per_thread, typename func_t>
static void launch_unfold_backward_kernel(int total_n_elems, func_t f) {
  const int total_work_block = n_threads * n_elems_per_thread;
  dim3 grid((total_n_elems + total_work_block - 1) / total_work_block);
  dim3 block(n_threads);

  unfold_backward_elementwise_kernel<n_threads, n_elems_per_thread, func_t>
    <<<grid, block>>>(total_n_elems, f);
}

template <typename scalar_t>
void unfold_backward_internal_kernel(
        scalar_t* __restrict__ grad_out_ptr,
  const scalar_t* __restrict__ grad_in_ptr,
  const int64_t* __restrict__ idx_dim_ptr,
  int64_t size,
  int64_t step,
  int64_t grad_in_dim_stride,
  int64_t grad_in_last_dim_stride,
  int64_t grad_in_dim_size,
  int64_t grad_out_dim_stride) 
{
  // For each index in grad_out find
  // the elements contributing to it and sum them up.
  // Note: the algorithm does not require any synchronization.
  auto loop = [=] __device__ (int i) {
    int64_t offsets[] = {i, i, 0};

    auto* __restrict__ grad_out_data = grad_out_ptr + offsets[0];
    auto* __restrict__ grad_in_data = grad_in_ptr + offsets[1];
    auto idx_dim = *(idx_dim_ptr + offsets[2]);

    // left_fold potentially intersecting with idx_dim
    // is either (idx_dim - size) / step or the next integer.
    int64_t left_fold_idx = (idx_dim > size) ? (idx_dim - size) / step : 0;
    if (!(left_fold_idx * step <= idx_dim && idx_dim < left_fold_idx * step + size)) {
      ++left_fold_idx;
    }

    auto right_fold_idx = idx_dim / step;
    right_fold_idx = (right_fold_idx >= grad_in_dim_size) ?
                     (grad_in_dim_size - 1) : right_fold_idx;

    for (auto fold_idx = left_fold_idx; fold_idx <= right_fold_idx; ++fold_idx) {
      auto idx_last_dim = idx_dim - fold_idx * step;
      *grad_out_data += grad_in_data[fold_idx * grad_in_dim_stride +
                                     idx_last_dim * grad_in_last_dim_stride];
    }
  };

  launch_unfold_backward_kernel<64, 4>(grad_in_dim_size, loop);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelem = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int64_t size = 2;
  const int64_t step = 1;
  const int64_t grad_in_dim_stride = 1;
  const int64_t grad_in_last_dim_stride = 1;
  const int64_t grad_in_dim_size = nelem;
  const int64_t grad_out_dim_stride = 1;

  using scalar_t = int;

  size_t input_size_bytes = grad_in_dim_size * sizeof(scalar_t);
  size_t output_size_bytes = grad_in_dim_size * sizeof(scalar_t);

  scalar_t *h_grad_in = (scalar_t*) malloc (input_size_bytes);
  scalar_t *h_grad_out = (scalar_t*) malloc (output_size_bytes);

  srand(123);
  for (int i = 0; i < grad_in_dim_size; i++) {
    h_grad_in[i] = rand() % 256;
  }

  const int64_t h_idx_dim = 0; // one dimension

  scalar_t *d_grad_in, *d_grad_out;
  cudaMalloc((void**)&d_grad_in, input_size_bytes);
  cudaMemcpy(d_grad_in, h_grad_in, input_size_bytes, cudaMemcpyHostToDevice); 

  int64_t *d_idx_dim;
  cudaMalloc((void**)&d_idx_dim, sizeof(int64_t));
  cudaMemcpy(d_idx_dim, &h_idx_dim, sizeof(int64_t), cudaMemcpyHostToDevice); 

  cudaMalloc((void**)&d_grad_out, output_size_bytes);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    unfold_backward_internal_kernel(d_grad_out,
                                    d_grad_in,
                                    d_idx_dim,
                                    size,
                                    step,
                                    grad_in_dim_stride,
                                    grad_in_last_dim_stride,
                                    grad_in_dim_size,
                                    grad_out_dim_stride);
  }
  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of unfold backward kernel: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(h_grad_out, d_grad_out, output_size_bytes, cudaMemcpyDeviceToHost); 

  bool ok = true;
  for (int i = 0; i < grad_in_dim_size; i++) {
    if (repeat * h_grad_in[i] != h_grad_out[i]) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
 
  cudaFree(d_grad_in);
  cudaFree(d_grad_out);
  cudaFree(d_idx_dim);

  free(h_grad_in);
  free(h_grad_out);

  return 0;
}
