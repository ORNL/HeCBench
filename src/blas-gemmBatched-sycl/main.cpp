#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>
#ifdef USE_ONEMATH
#include <oneapi/math.hpp>
namespace math_ns = oneapi::math;
#else
#include <oneapi/mkl.hpp>
namespace math_ns = oneapi::mkl;
#endif
#include "reference.h"

using namespace std;

template <typename T>
void gemmBatched(
  int lower,
  int upper,
  int num,
  int reps,
  int verbose) try
{
  if(verbose) cout << "initializing inputs" << endl;
  size_t matrices_size = upper * upper * num * sizeof(T);
  size_t vectors_size = upper * num * sizeof(T);

  T *matrices = (T*)malloc(matrices_size);
  assert(matrices);

  T *vectors = (T*)malloc(vectors_size);
  assert(vectors);

  T *result = (T*)malloc(vectors_size);
  assert(result);

  T *result_ref = (T*)malloc(vectors_size);
  assert(result_ref);

  srand48(48);
  for(int i = 0; i < num * upper * upper; i++)
    matrices[i] = static_cast<T>(drand48());

  for(int i = 0; i < num * upper; i++)
    vectors[i] = static_cast<T>(drand48());

  if(verbose) cout << "allocating device variables" << endl;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // allocate input space on device
  T *devMatrices = (T *)sycl::malloc_device(matrices_size, q);
  assert(devMatrices != nullptr);

  T *devVectors = (T *)sycl::malloc_device(vectors_size, q);
  assert(devVectors != nullptr);

  // allocate result space on device
  T *devResult = (T *)sycl::malloc_device(vectors_size, q);
  assert(devResult != nullptr);

  if(verbose) cout << "copying data to device" << endl;
  // copy data to device
  q.memcpy(devMatrices, matrices, matrices_size).wait();
  q.memcpy(devVectors, vectors, vectors_size).wait();

  // create lists of device pointers to inputs and outputs
  T **AList = 0, **BList = 0, **CList = 0;

  AList = (T**)malloc(num * sizeof(T*));
  BList = (T**)malloc(num * sizeof(T*));
  CList = (T**)malloc(num * sizeof(T*));

  int lda = upper, // lda >= max(1,m)
      ldb = upper, // ldb >= max(1,k)
      ldc = upper; // ldc >= max(1,m)

  const T alpha = 1.0f, beta = 0.0f;
  for(int i = 0; i < num; i++){
    // each array of dim. lda x k
    AList[i] = devMatrices + upper * upper * i;
    // each array of dim. ldb x n
    BList[i] = devVectors + upper * i;
    // each array of dim. ldc x n
    CList[i] = devResult + upper * i;
  }

  // copy pointer lists to device
  T **devAList, **devBList, **devCList;
  devAList = sycl::malloc_device<T *>(num, q);
  assert(devAList != nullptr);

  devBList = sycl::malloc_device<T *>(num, q);
  assert(devBList != nullptr);

  devCList = sycl::malloc_device<T *>(num, q);
  assert(devCList != nullptr);

  q.memcpy(devAList, AList, num * sizeof(T *)).wait();
  q.memcpy(devBList, BList, num * sizeof(T *)).wait();
  q.memcpy(devCList, CList, num * sizeof(T *)).wait();


  /* perform <num> <size x size> x <size x 1> multiplications
     with distinct matrices
   */
  struct param_t {
    math_ns::transpose transpose_info[2];
    T value_info[2];
    std::int64_t size_info[3];
    std::int64_t ld_info[3];
    std::int64_t groupsize_info;
  };

  param_t *p = (param_t *)std::malloc(sizeof(param_t));
  p->transpose_info[0] = math_ns::transpose::nontrans;
  p->transpose_info[1] = math_ns::transpose::nontrans;
  p->value_info[0] = alpha;
  p->value_info[1] = beta;
  p->ld_info[0] = lda;
  p->ld_info[1] = ldb;
  p->ld_info[2] = ldc;
  p->groupsize_info = num;

  for(int size = lower; size <= upper; size++){
    if(verbose) cout << "running with <size x size> x <size x 1> " << size << endl;
    double sum = 0.0;
    const int m = size, n = 1, k = size;
    p->size_info[0] = m;
    p->size_info[1] = n;
    p->size_info[2] = k;
    for(int rep = 0; rep <= reps; rep++){
      auto start = std::chrono::steady_clock::now();

      math_ns::blas::column_major::gemm_batch(
        q, p->transpose_info, p->transpose_info + 1,
        p->size_info, p->size_info + 1,
        p->size_info + 2, p->value_info,
        const_cast<const T**>(devAList), p->ld_info,
        const_cast<const T**>(devBList), p->ld_info + 1,
        p->value_info + 1, devCList,
        p->ld_info + 2, 1, &(p->groupsize_info)).wait();

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      auto elapsed = time * 1e-3;

      if (rep != 0) sum += elapsed;

      if(verbose)
        cout << "size " << size << ": " << elapsed << " us; "
             << elapsed / num << " us per operation" << endl;
    }
    cout << "size " << size << " average execution time: " << sum/reps << " us; "
         << sum / reps / num << " us per operation; "
         << "floating-point operations per second: ";
    performance(m, n, k, 1e3 * (sum / reps / num));

    // verify double precision operations
    if constexpr (std::is_same_v<T, double>) {
      q.memcpy(result, devResult, vectors_size).wait();
      gemmBatched_ref (num, upper, upper, 1, m, k, n, alpha, beta,
                       matrices, lda, vectors, ldb, result_ref, ldc);

      int ok = 1;
      for (int i = 0; i < num; i++) {
      for (int j = 0; j < m; j++) {
        if (abs(result[i*upper+j] - result_ref[i*upper+j]) > 1e-6) {
          cout << "Mismatch at batch index " << i << ": " << result[i*upper+j] << "!="
               << result_ref[i*upper+j] << endl;
          ok = 0;
          break;
        }
      }}

      printf("%s\n", ok ? "PASS" : "FAIL");
    }
  }

  sycl::free(devMatrices, q);
  sycl::free(devVectors, q);
  sycl::free(devResult, q);
  sycl::free(devAList, q);
  sycl::free(devBList, q);
  sycl::free(devCList, q);

  free(p);
  free(matrices);
  free(vectors);
  free(result);
  free(result_ref);
  free(AList);
  free(BList);
  free(CList);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int main(int argc, char **argv) {

  int status;
  int lower = 2;    // lower bound to the matrix dimension
  int upper = 100;  // upper bound to the matrix dimension
  int num = 25000;  // batch size
  int reps = 10;
  int verbose = 0;

  while((status = getopt(argc, argv, "l:u:n:r:v")) != -1){
    switch(status){
    case 'l':
      lower = strtoul(optarg, 0, 0);
      break;
    case 'u':
      upper = strtoul(optarg, 0, 0);
      break;
    case 'n':
      num = strtoul(optarg, 0, 0);
      break;
    case 'r':
      reps = strtoul(optarg, 0, 0);
      break;
    case 'v':
      verbose = 1;
      break;
    default:
      cerr << "invalid argument: " << status << endl;
      exit(1);
    }
  }

  cout << "running with" << " lower: " << lower << " upper: " << upper
       << " num: " << num << " reps: " << reps << endl;

  cout << ">>>>>>>>>>>>>>> Half precision gemmBatched >>>>>>>>>>>>>>> " << endl;
  gemmBatched<sycl::half>(lower, upper, num, reps, verbose);
  cout << ">>>>>>>>>>>>>>> Single precision gemmBatched >>>>>>>>>>>>>>> " << endl;
  gemmBatched<float>(lower, upper, num, reps, verbose);
  cout << ">>>>>>>>>>>>>>> Double precision gemmBatched >>>>>>>>>>>>>>> " << endl;
  gemmBatched<double>(lower, upper, num, reps, verbose);

  return 0;
}
