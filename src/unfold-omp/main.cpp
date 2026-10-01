#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <omp.h>

template <int n_threads, int n_elems_per_thread, typename func_t>
void unfold_backward_elementwise_kernel(int grid, int total_n_elems, func_t f) {
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  #pragma omp target teams distribute parallel for collapse(2) \
   num_teams(grid) num_threads(n_threads)
  for (int bid = 0; bid < grid; bid++) {
    for (int tid = 0; tid < n_threads; tid++) {
      int idx = bid * total_work_block + tid;
      #pragma unroll
      for (int i = 0; i < n_elems_per_thread; ++i) {
        if (idx < total_n_elems) {
          f(idx);
          idx += n_threads;
        }
      }
    }
  }
}

template <int n_threads, int n_elems_per_thread, typename func_t>
static void launch_unfold_backward_kernel(int total_n_elems, func_t f) {
  const int total_work_block = n_threads * n_elems_per_thread;
  int grid = ((total_n_elems + total_work_block - 1) / total_work_block);
  unfold_backward_elementwise_kernel<n_threads, n_elems_per_thread, func_t>
    (grid, total_n_elems, f);
}

template <typename scalar_t>
void unfold_backward_internal_kernel(
        scalar_t* __restrict__ grad_out_ptr,
  const scalar_t* __restrict__ grad_in_ptr,
  const int64_t idx_dim,
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
  auto loop = [=] (int i) {
    int64_t offsets[] = {i, i};

    auto* __restrict__ grad_out_data = grad_out_ptr + offsets[0];
    auto* __restrict__ grad_in_data = grad_in_ptr + offsets[1];

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
  memset(h_grad_out, 0, output_size_bytes);

  srand(123);
  for (int i = 0; i < grad_in_dim_size; i++) {
    h_grad_in[i] = rand() % 256;
  }

  const int64_t h_idx_dim = 0; // one dimension

  #pragma omp target data map(to: h_grad_in[0:grad_in_dim_size], h_idx_dim) \
                          map(tofrom: h_grad_out[0:grad_in_dim_size])

  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      unfold_backward_internal_kernel(h_grad_out,
                                      h_grad_in,
                                      h_idx_dim,
                                      size,
                                      step,
                                      grad_in_dim_stride,
                                      grad_in_last_dim_stride,
                                      grad_in_dim_size,
                                      grad_out_dim_stride);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of unfold backward kernel: %f (us)\n", (time * 1e-3f) / repeat);
  }

  bool ok = true;
  for (int i = 0; i < grad_in_dim_size; i++) {
    if (repeat * h_grad_in[i] != h_grad_out[i]) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(h_grad_in);
  free(h_grad_out);

  return 0;
}
