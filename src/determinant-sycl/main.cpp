#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <oneapi/mkl/lapack.hpp>
#include <chrono>
#include <cmath>
#include <stdio.h>
#include <sycl/sycl.hpp>

/*************************/
/* STRIDED RANGE FUNCTOR */
/*************************/
template <typename Iterator>
class strided_range
{
  public:
    typedef typename std::iterator_traits<Iterator>::difference_type
      difference_type;

    struct stride_functor {
      difference_type stride;

      stride_functor(difference_type stride)
        : stride(stride) {}

      difference_type operator()(const difference_type& i) const
      {
        return stride * i;
      }
    };

    typedef typename oneapi::dpl::counting_iterator<difference_type> CountingIterator;
    typedef typename oneapi::dpl::transform_iterator<CountingIterator, stride_functor> TransformIterator;
    typedef typename oneapi::dpl::permutation_iterator<Iterator, TransformIterator> PermutationIterator;

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

  float det = 0;

#ifdef DEBUG
  // Variable holding status of calculations
  std::int64_t info = 0;

  // Asynchronous error handler
  auto error_handler = [&] (sycl::exception_list exceptions) {
    for (auto const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch(oneapi::mkl::lapack::exception const& e) {
        // Handle LAPACK related exceptions happened during asynchronous call
        info = e.info();
        std::cout << "Unexpected exception caught during asynchronous LAPACK operation:\n"
                  << e.what() << "\ninfo: " << e.info() << std::endl;
      } catch(sycl::exception const& e) {
        // Handle not LAPACK related exceptions happened during asynchronous call
        std::cout << "Unexpected exception caught during asynchronous operation:\n" << e.what() << std::endl;
        info = -1;
      }
    }
  };

  try {
#endif

  sycl::queue q(
#ifdef USE_GPU
    sycl::gpu_selector_v,
#else
    sycl::cpu_selector_v,
#endif
#ifdef DEBUG
    error_handler,
#endif
    sycl::property::queue::in_order());

    // --- Setting the device matrix and moving the host matrix to the device
    sycl::buffer<float, 1> d_A (Nrows * Ncols);

    auto policy = oneapi::dpl::execution::make_device_policy(q);

    // --- CHOLESKY initialization
    int scratchpad_size = oneapi::mkl::lapack::potrf_scratchpad_size<float>(
        q, oneapi::mkl::uplo::lower, Nrows, Nrows);

    // --- POTRF execution
    sycl::buffer<float, 1> scratchpad(scratchpad_size);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        auto acc = d_A.get_access<sycl::access::mode::write>(cgh);
        cgh.copy(h_A, acc);
      });

      oneapi::mkl::lapack::potrf(
          q, oneapi::mkl::uplo::lower, Nrows, d_A, Nrows, scratchpad, scratchpad_size);

      // --- Strided reduction of the elements of d_A:
      // calculating the product of the diagonal of the Cholesky factorization
      strided_range pos(oneapi::dpl::begin(d_A), oneapi::dpl::end(d_A), STRIDE);

      det = oneapi::dpl::reduce(policy, pos.begin(), pos.end(), 1.f, std::multiplies<float>());
    }

#ifdef DEBUG
    q.wait_and_throw();
#else
    q.wait();
#endif
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);

    float result = det * det;
    printf("determinant = %f\n", result);
    printf("%s\n", (std::fabs(result - 2048) < 0.02f) ? "PASS" : "FAIL");

#ifdef DEBUG
  } catch(oneapi::mkl::lapack::exception const& e) {
    // Handle LAPACK related exceptions happened during synchronous call
    std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\nreason: "
              << e.what() << "\ninfo: " << e.info() << std::endl;
    info = e.info();
  } catch(sycl::exception const& e) {
    // Handle not LAPACK related exceptions happened during synchronous call
    std::cout << "Unexpected exception caught during synchronous call to SYCL API:\n" << e.what() << std::endl;
    info = -1;
  }

  std::cout << "oneapi::mkl::lapack::potrf " << ((info == 0) ? "ran OK" : "FAILED") << std::endl;
#endif

  return 0;
}
