#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <sycl/sycl.hpp>

template <int n_threads, int n_elems_per_thread, typename func_t>
void unfold_backward_elementwise_kernel(int total_n_elems, func_t f,
                                        const sycl::nd_item<1> &item) {
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  int idx = total_work_block * item.get_group(0) + item.get_local_id(0);
  #pragma unroll
  for (int i = 0; i < n_elems_per_thread; ++i) {
    if (idx < total_n_elems) {
      f(idx);
      idx += n_threads;
    }
  }
}

template <int n_threads, int n_elems_per_thread, typename func_t>
static void launch_unfold_backward_kernel(sycl::queue &q, int total_n_elems, func_t f) {
  const int total_work_block = n_threads * n_elems_per_thread;

  sycl::range<1> lws (n_threads);
  sycl::range<1> gws ((total_n_elems + total_work_block - 1) /
                      total_work_block * n_threads);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      unfold_backward_elementwise_kernel
        <n_threads, n_elems_per_thread, func_t>(total_n_elems, f, item);
    });
  });
}

template <typename scalar_t>
void unfold_backward_internal_kernel(
  sycl::queue &q,
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
  auto loop = [=] (int i) {
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

  launch_unfold_backward_kernel<64, 4>(q, grad_in_dim_size, loop);
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  scalar_t *d_grad_in, *d_grad_out;
  d_grad_in = (scalar_t *)sycl::malloc_device(input_size_bytes, q);
  q.memcpy(d_grad_in, h_grad_in, input_size_bytes);

  int64_t *d_idx_dim = sycl::malloc_device<int64_t>(1, q);
  q.memcpy(d_idx_dim, &h_idx_dim, sizeof(int64_t));

  d_grad_out = (scalar_t *)sycl::malloc_device(output_size_bytes, q);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    unfold_backward_internal_kernel(q,
                                    d_grad_out,
                                    d_grad_in,
                                    d_idx_dim,
                                    size,
                                    step,
                                    grad_in_dim_stride,
                                    grad_in_last_dim_stride,
                                    grad_in_dim_size,
                                    grad_out_dim_stride);
  }
  q.wait();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of unfold backward kernel: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(h_grad_out, d_grad_out, output_size_bytes).wait();

  bool ok = true;
  for (int i = 0; i < grad_in_dim_size; i++) {
    if (repeat * h_grad_in[i] != h_grad_out[i]) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_grad_in, q);
  sycl::free(d_grad_out, q);
  sycl::free(d_idx_dim, q);

  free(h_grad_in);
  free(h_grad_out);

  if (!ok) return 1;
  return 0;
}
