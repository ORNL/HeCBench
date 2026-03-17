#include <chrono>
#include <cmath>
#include <stdio.h>

#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/device_vector.h>

/*************************/
/* STRIDED RANGE FUNCTOR */
/*************************/
template <typename Iterator>
class strided_range
{
  public:

  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct stride_functor
  {
    difference_type stride;

    stride_functor(difference_type stride)
      : stride(stride) {}

    __host__ __device__
    difference_type operator()(const difference_type& i) const
    {
      return stride * i;
    }
  };

  typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
  typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  strided_range(Iterator first, Iterator last, difference_type stride)
    : first(first), last(last), stride(stride) {}

  iterator begin(void) const
  {
    return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
  }

  iterator end(void) const
  {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }

  protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // --- Setting the host, Nrows x Ncols matrix
  const int Nrows = 11;
  const int Ncols = 11;

  float h_A[Nrows*Ncols] = {
     2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
    -2.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    -2.,  0.,  6.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
    -2.,  0.,  2.,  8.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
    -2.,  0.,  2.,  4.,  10., 10., 10., 10., 10., 10., 10.,
    -2.,  0.,  2.,  4.,  10., 12., 12., 12., 12., 12., 12.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 14., 14., 14., 14.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 16., 16., 16.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 18., 18.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 20., 20.,
    -2.,  0.,  2.,  4.,  10., 12., 14., 16., 18., 20., 22.
  };

  const int STRIDE = Nrows + 1;

  // --- Setting the device matrix and moving the host matrix to the device
  float *d_A;
  cudaMalloc(&d_A, Nrows * Ncols * sizeof(float));

  // --- input/output parameters/arrays
  int work_size = 0;
  int *devInfo;
  cudaMalloc(&devInfo, sizeof(int));

  // --- solver initialization
  cusolverDnHandle_t solver_handle;
  cusolverDnCreate(&solver_handle);
  cusolverStatus_t status;

  // --- CHOLESKY initialization
  status = cusolverDnSpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, d_A, Nrows, &work_size);
#ifdef DEBUG
  if (status != CUSOLVER_STATUS_SUCCESS) 
    printf("Unsuccessful cusolverDnSpotrf_bufferSize call\n\n");
#endif

  float *work;
  cudaMalloc(&work, work_size * sizeof(float));

  float det = 0;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(float), cudaMemcpyHostToDevice);

    status = cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, Nrows, d_A, Nrows, work, work_size, devInfo);
#ifdef DEBUG
    int devInfo_h = 0;
    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != CUSOLVER_STATUS_SUCCESS || devInfo_h != 0)
      printf("Unsuccessful cusolverDnSpotrf execution\n\n");
#endif

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_A);
    typedef thrust::device_vector<float>::iterator Iterator;

    // --- Strided reduction of the elements of d_A:
    // calculating the product of the diagonal of the Cholesky factorization
    strided_range<Iterator> pos(dev_ptr, dev_ptr + Nrows * Ncols, STRIDE);

    det = thrust::reduce(pos.begin(), pos.end(), 1.f, thrust::multiplies<float>());
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);

  float result = det * det;
  printf("determinant = %f\n", result);
  printf("%s\n", (std::fabs(result - 2048) < 0.02f) ? "PASS" : "FAIL");

  status = cusolverDnDestroy(solver_handle);
#ifdef DEBUG
  if (status != CUSOLVER_STATUS_SUCCESS) 
    printf("Unsuccessful cusolverDnDestroy call\n\n");
#endif
  cudaFree(d_A);
  cudaFree(work);
  cudaFree(devInfo);

  return 0;
}
