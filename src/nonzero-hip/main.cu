#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

// Reference
// https://pytorch.org/docs/stable/generated/torch.nonzero.html#torch.nonzero

#define MAX_DIMS 2

template<typename index_t>
struct TensorDims {
  index_t sizes[MAX_DIMS];
};

template <typename index_t>
__global__
void write_indices(int64_t* inp, TensorDims<index_t> dims, int ndim, index_t nzero)
{
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
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
  __host__ __device__ __forceinline__
  bool operator()(const T &a) const { return a != (T)0; }
};

template <typename scalar_t>
void nonzero (int nrows, int ncols, int repeat) {
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

  for (int n = 0; n < repeat; n++) {

    int64_t r_nzeros = 0;

    // Ensure non-zero element(s)
    do {
      for (int i = 0; i < num_items; i++) {
        h_in[i] = (scalar_t) dist(gen); 
        if (h_in[i] != (scalar_t)0) r_nzeros++;
      }
    } while (r_nzeros == 0);

    scalar_t *d_in;
    hipMalloc((void**)&d_in, elem_size_bytes);
    hipMemcpy(d_in, h_in, elem_size_bytes, hipMemcpyHostToDevice);

    // Number of non-zeros computed by a device
    int64_t h_nzeros;
    int64_t *d_nzeros;
    hipMalloc((void**)&d_nzeros, sizeof(int64_t));

    // Time the sum reduction on a device
    auto start = std::chrono::steady_clock::now();

    NonZero<scalar_t> conversion_op;

    // Create an iterator wrapper
    hipcub::TransformInputIterator<bool, NonZero<scalar_t>, scalar_t*> itr (d_in, conversion_op);

    // Determine temporary device storage requirements
    void     *d_temp_storage;
    size_t   temp_storage_bytes = 0;

    hipcub::DeviceReduce::Sum(nullptr, temp_storage_bytes, itr, d_nzeros, num_items);

    hipMalloc(&d_temp_storage, temp_storage_bytes);

    hipcub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_nzeros, num_items);

    hipFree(d_temp_storage);

    hipMemcpy(&h_nzeros, d_nzeros, sizeof(int64_t), hipMemcpyDeviceToHost);

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

      int64_t *d_out;
      hipMalloc((void**)&d_out, d_out_size_bytes);

      // Time the index operations on a device
      auto start = std::chrono::steady_clock::now();

      hipcub::CountingInputIterator<int64_t> counting_itr(0);

      hipcub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, counting_itr, itr, d_out, d_nzeros, num_items);

      hipMalloc(&d_temp_storage, temp_storage_bytes);

      hipcub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, counting_itr, itr, d_out, d_nzeros, num_items);

      TensorDims<int64_t> out_dims;
      for (int i = 0; i < in_ndims; i++) {
        out_dims.sizes[i] = in_sizes[i];
      }

      const int nthreads = 256;
      const int nblocks = (h_nzeros + nthreads - 1) / nthreads;

      write_indices<int64_t><<<nblocks, nthreads>>>(d_out, out_dims, in_ndims, h_nzeros);

      hipFree(d_temp_storage);

      hipDeviceSynchronize();
      auto end = std::chrono::steady_clock::now();
      idx_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
      hipMemcpy(h_out, d_out, d_out_size_bytes, hipMemcpyDeviceToHost);

      hipFree(d_out);
  
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

    hipFree(d_nzeros);
    hipFree(d_in);

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

    printf("=========== Data type is FP64 ==========\n");
    nonzero<double> (nrows, ncols, repeat);
  }

  return 0;
}
