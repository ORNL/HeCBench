#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <random>

// Reference
// https://pytorch.org/docs/stable/generated/torch.nonzero.html#torch.nonzero

#define MAX_DIMS 2

template<typename index_t>
struct TensorDims {
  index_t sizes[MAX_DIMS];
};

template <typename index_t>
class k_indices;

// https://github.com/oneapi-src/oneDPL/issues/814
template<typename Policy, typename InputIt, typename FlaggedIt, typename OutputIt>
void flagged(Policy&& policy,
             InputIt it,
             FlaggedIt flagged,
             OutputIt out,
             //std::int64_t *num_copied,
             std::int64_t size)
{
  auto zip_b = oneapi::dpl::make_zip_iterator(it, flagged);
  auto zip_e = oneapi::dpl::make_zip_iterator(it + size, flagged + size);
  auto out_it = oneapi::dpl::make_zip_iterator(out, oneapi::dpl::discard_iterator{});

  auto end_it = std::copy_if(std::forward<Policy>(policy), zip_b, zip_e, out_it, [](auto const & x) {
      return std::get<1>(x);
  });

  // the value of num_copied is not used
  //q.fill(num_copied, std::distance(out_it, end_it), 1).wait();
}

template <typename index_t>
void write_indices(int64_t* inp, TensorDims<index_t> dims, int ndim, index_t nzero,
                   sycl::nd_item<1> &item)
{
  auto index = item.get_global_id(0);
  if (index < nzero) {
    index_t div = 1;
    int64_t idx_flat = inp[index];
    #pragma unroll
    for (int dim = MAX_DIMS; dim >= 0; dim--) {
      if (dim > ndim - 1) continue;
      auto dim_size = dims.sizes[dim];
      inp[index + dim * nzero] = (idx_flat / div) % dim_size;
      div *= dim_size;
    }
  }
}

template <typename T>
struct NonZero
{
  inline bool operator()(const T &a) const { return a != (T)0; }
};

template <typename scalar_t> void nonzero(int nrows, int ncols, int repeat) {

  // The total number of dimensions is a i32 number and
  // the size of each dimension is a i64 number
  const int in_ndims = MAX_DIMS;
  int64_t in_sizes[MAX_DIMS] = {nrows, ncols};

  // Total number of elements
  int64_t num_items = 1;
  for (int i = 0; i < in_ndims; i++) {
    num_items *= in_sizes[i];
  }
  int64_t elem_size_bytes = num_items * sizeof(scalar_t);

  std::mt19937 gen (19937);

  std::uniform_int_distribution<> dist (-1, 1);

  scalar_t *h_in = (scalar_t*) malloc (elem_size_bytes);

  bool ok = true;
  long sum_time = 0;
  long idx_time = 0;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  auto policy = oneapi::dpl::execution::device_policy(q);

  for (int n = 0; n < repeat; n++) {

    int64_t r_nzeros = 0;

    // Ensure non-zero element(s)
    do {
      for (int i = 0; i < num_items; i++) {
        h_in[i] = (scalar_t) dist(gen);
        if (h_in[i] != (scalar_t)0) r_nzeros++;
      }
    } while (r_nzeros == 0);

    scalar_t *d_in = sycl::malloc_device<scalar_t>(num_items, q);
    q.memcpy(d_in, h_in, elem_size_bytes).wait();

    // Number of non-zeros computed by a device
    int64_t h_nzeros;

    // Time the sum reduction on a device
    auto start = std::chrono::steady_clock::now();

    NonZero<scalar_t> conversion_op;

    // Create an iterator wrapper
    oneapi::dpl::transform_iterator<scalar_t *, NonZero<scalar_t>> itr(
        d_in, conversion_op);

    h_nzeros = oneapi::dpl::reduce(policy, itr, itr + num_items, (int64_t)0);

    auto end = std::chrono::steady_clock::now();
    sum_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    if (h_nzeros != r_nzeros) {

      printf("Number of non-zero elements mismatch: %ld != %ld (expected)\n",
             h_nzeros, r_nzeros);
      ok = false;

    } else {

      // Output size is z x n (z: number of non-zero and n is the dimension of input)
      int64_t d_out_size = h_nzeros * in_ndims;
      int64_t d_out_size_bytes = d_out_size * sizeof(int64_t);

      int64_t *h_out = (int64_t*) malloc (d_out_size_bytes);

      int64_t *d_out = sycl::malloc_device<int64_t>(d_out_size, q);

      // Time the index operations on a device
      auto start = std::chrono::steady_clock::now();

      oneapi::dpl::counting_iterator<int64_t> counting_itr(0);

      flagged(policy, counting_itr, itr, d_out, num_items);

      TensorDims<int64_t> out_dims;
      for (int i = 0; i < in_ndims; i++) {
        out_dims.sizes[i] = in_sizes[i];
      }

      const int nthreads = 256;
      const int nblocks = (h_nzeros + nthreads - 1) / nthreads;
      sycl::range<1> gws (nblocks * nthreads);
      sycl::range<1> lws (nthreads);

      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class k_indices<scalar_t>>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          write_indices<int64_t>(d_out, out_dims, in_ndims, h_nzeros, item);
        });
      }).wait();

      auto end = std::chrono::steady_clock::now();
      idx_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      q.memcpy(h_out, d_out, d_out_size_bytes).wait();

      sycl::free(d_out, q);

      // verify the results

      // The output is a tuple of 1-D tensors, one for each dimension in input,
      // each containing the indices (in that dimension) of all non-zero elements of input .
      // If input has n dimensions, then the resulting tuple contains n tensors of size z,
      // where z is the total number of non-zero elements in the input tensor.

      int64_t cnt_nzero = 0;

      for (int i = 0; i < h_nzeros; i++) {
        if (in_ndims == 1) {
          if (h_in[h_out[i]] != 0)
            cnt_nzero++;
        }
        if (in_ndims == 2) {
          if (h_in[in_sizes[1] * h_out[i] + h_out[h_nzeros + i]] != 0)
            cnt_nzero++;
        }
      }

      ok = cnt_nzero == h_nzeros;

      free(h_out);
    }

    sycl::free(d_in, q);

    if (!ok) break;
  }

  free(h_in);

  printf("Average time for sum reduction: %lf (us)\n",
         sum_time * 1e-3 / repeat);

  printf("Average time for write index operations: %lf (us)\n",
         idx_time * 1e-3 / repeat);

  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }

  int nrows = atoi(argv[1]);
  int ncols = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  if (nrows <= 0) nrows = 1;
  if (ncols <= 0) ncols = 1;

  // Warmup may be needed for more accurate performance measurement
  for (int w = 0; w < 2; w++) {
    printf("=========== Data type is I8 ==========\n");
    nonzero<int8_t> (nrows, ncols, repeat);

    printf("=========== Data type is I16 ==========\n");
    nonzero<int16_t> (nrows, ncols, repeat);

    printf("=========== Data type is I32 ==========\n");
    nonzero<int32_t> (nrows, ncols, repeat);

    printf("=========== Data type is FP32 ==========\n");
    nonzero<float> (nrows, ncols, repeat);

    //printf("=========== Data type is FP64 ==========\n");
    //nonzero<double> (nrows, ncols, repeat);
  }

  return 0;
}
