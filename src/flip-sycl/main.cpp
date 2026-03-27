#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <sycl/sycl.hpp>
#include "reference.h"

// Example
// https://pytorch.org/docs/stable/generated/torch.flip.html

template <typename scalar_t>
void flip_kernel(
    sycl::nd_item<1> &item,
    const scalar_t* in_tensor,
          scalar_t* out_tensor,
    int64_t  n,
    const int64_t* flip_dims,
    const int64_t  flip_dims_size,
    const int64_t* strides,
    const int64_t* strides_contiguous,
    const int64_t* shape,
    const int64_t  total_dims)
{
  int64_t linear_index = item.get_global_id(0);

  if (linear_index >= n) return;

  int64_t cur_indices = linear_index;
  int64_t rem = 0;
  int64_t dst_offset = 0;

  for (int64_t i = 0; i < total_dims; i++) {
    int64_t temp = cur_indices;
    cur_indices = cur_indices / strides_contiguous[i];
    rem = temp - cur_indices * strides_contiguous[i];
    for (int64_t j = 0; j < flip_dims_size; j++) {
      // flip the indices if it is in flip_dims
      if (i == flip_dims[j]) {
        cur_indices = shape[i] - 1 - cur_indices;
      }
    }
    dst_offset += cur_indices * strides[i];
    cur_indices = rem;
  }
  out_tensor[linear_index] = in_tensor[dst_offset];
}

// display the values of a property in a tensor
void print_property (const char* name, std::vector<int64_t> p)
{
  printf("%s: ( ", name);
  for (uint64_t i = 0; i < p.size(); i++) {
    printf("%lu ", p[i]);
  }
  printf(")\n");
}

template <typename scalar_t>
void flip (const int64_t num_dims, const int64_t num_flip_dims,
           const int32_t dim_size, const int32_t repeat)
{
  std::vector<int64_t> flip;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;

  for (int64_t i = 0; i < num_dims; i++) {
#ifdef EXAMPLE
    shape.push_back(2);
#else
    shape.push_back(dim_size);
#endif
  }

  int64_t n = 1;
  for (int64_t i = 0; i < num_dims; i++) {
    n = n * shape[i];
  }

  for (int64_t i = 0; i < num_flip_dims; i++) {
    flip.push_back(i);
  }

  stride.push_back(shape[1] * shape[2]);
  stride.push_back(shape[2]);
  stride.push_back(1);

  print_property("shape", shape);
  print_property("flip_dims", flip);
  print_property("stride", stride);

  int64_t dims_bytes = num_dims * sizeof(int64_t);
  int64_t flip_dims_bytes = num_flip_dims * sizeof(int64_t);
  int64_t input_size_bytes = n * sizeof(scalar_t);
  int64_t output_size_bytes = input_size_bytes;

  scalar_t *input = (scalar_t*) malloc (input_size_bytes);

  for (int i = 0; i < n; i++) {
    input[i] = (scalar_t) i;
  }

  scalar_t *output = (scalar_t*) malloc(output_size_bytes);
  scalar_t *output_ref = (scalar_t*) malloc(output_size_bytes);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  scalar_t *d_input, *d_output;
  d_input = sycl::malloc_device<scalar_t>(n, q);
  q.memcpy(d_input, input, input_size_bytes);

  d_output = sycl::malloc_device<scalar_t>(n, q);

  int64_t *d_flip_dims, *d_shape, *d_strides, *d_strides_contiguous;

  d_flip_dims = sycl::malloc_device<int64_t>(num_flip_dims, q);
  q.memcpy(d_flip_dims, flip.data(), flip_dims_bytes);

  d_shape = sycl::malloc_device<int64_t>(num_dims, q);
  q.memcpy(d_shape, shape.data(), dims_bytes);

  d_strides = sycl::malloc_device<int64_t>(num_dims, q);
  q.memcpy(d_strides, stride.data(), dims_bytes);

  d_strides_contiguous = sycl::malloc_device<int64_t>(num_dims, q);
  q.memcpy(d_strides_contiguous, stride.data(), dims_bytes);

  const int threadsPerBlock = 256;

  sycl::range<1> gws ((n + threadsPerBlock - 1) / threadsPerBlock * threadsPerBlock);
  sycl::range<1> lws (threadsPerBlock);

  auto kFn = [=] (sycl::nd_item<1> item) {
      flip_kernel<scalar_t>(
        item,
        d_input,
        d_output,
        n,
        d_flip_dims,
        num_flip_dims,
        d_strides,
        d_strides_contiguous,
        d_shape,
        num_dims);
   };

  // warmup and verify
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), kFn);
  });

  flip_kernel_cpu<scalar_t>(
    input, output_ref, n, flip.data(), num_flip_dims,
    stride.data(), stride.data(), shape.data(), num_dims);

  q.memcpy(output, d_output, output_size_bytes).wait();
  int error = memcmp(output, output_ref, output_size_bytes);
  printf("%s\n", error ? "FAIL" : "PASS");

#ifdef EXAMPLE
  for (int i = 0; i < n; i++) {
    printf("%f ", output[i]);
  }
  printf("\n");
#endif

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), kFn);
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the flip kernel: %f (ms)\n", (time * 1e-6f) / repeat);

  free(input);
  free(output);
  free(output_ref);
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  sycl::free(d_flip_dims, q);
  sycl::free(d_shape, q);
  sycl::free(d_strides, q);
  sycl::free(d_strides_contiguous, q);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of dimensions> <size of each dimension> <repeat>\n", argv[0]);
    return 1;
  }

  const int64_t num_dims = atoi(argv[1]);
  const int64_t dim_size = atoi(argv[2]);
  const int32_t repeat = atoi(argv[3]);

#ifdef EXAMPLE
  const int64_t num_flip_dims = 2;
#else
  const int64_t num_flip_dims = num_dims;
#endif

  printf("=========== Data type is FP32 ==========\n");
  flip<float>(num_dims, num_flip_dims, dim_size, repeat);

  printf("=========== Data type is FP64 ==========\n");
  flip<double>(num_dims, num_flip_dims, dim_size, repeat);

  return 0;
}
