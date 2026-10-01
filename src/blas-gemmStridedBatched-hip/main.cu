#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <type_traits> // is_same
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "reference.h"

using namespace std;

template <typename T>
void gemmBatched(
  int lower,
  int upper,
  int num,
  int reps,
  int verbose) 
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

  hipError_t cudaStat;
  hipblasStatus_t stat;
  hipblasHandle_t handle;

  stat = hipblasCreate(&handle);
  if(stat != HIPBLAS_STATUS_SUCCESS){
    cerr << "hipblas init failed" << endl;
    exit(1);
  }

  if(verbose) cout << "allocating device variables" << endl;

  // allocate input space on device
  T *devMatrices;
  cudaStat = hipMalloc((void**)&devMatrices, matrices_size);
  assert(!cudaStat);

  T *devVectors;
  cudaStat = hipMalloc((void**)&devVectors, vectors_size);
  assert(!cudaStat);

  // allocate result space on device
  T *devResult;
  cudaStat = hipMalloc((void**)&devResult, vectors_size);

  assert(!cudaStat);

  if(verbose) cout << "copying data to device" << endl;
  // copy data to device
  cudaStat = 
    hipMemcpy(devMatrices, matrices, matrices_size, hipMemcpyHostToDevice);

  assert(!cudaStat);
  
  cudaStat = 
    hipMemcpy(devVectors, vectors, vectors_size, hipMemcpyHostToDevice);

  assert(!cudaStat);

  int lda = upper, // lda >= max(1,m)
      ldb = upper, // ldb >= max(1,k)
      ldc = upper; // ldc >= max(1,m)

  const T alpha = 1.0f, beta = 0.0f;

  /* perform <num> <size x size> x <size x 1> multiplications 
     with distinct matrices
   */
#define GEMM_BATCHED_PARAMETERS handle,              \
                                HIPBLAS_OP_N,        \
                                HIPBLAS_OP_N,        \
                                m, n, k,             \
                                &alpha,              \
                                devMatrices,         \
                                lda,                 \
                                upper * upper,       \
                                devVectors,          \
                                ldb,                 \
                                upper,               \
                                &beta,               \
                                devResult,           \
                                ldc,                 \
                                upper,               \
                                num

  for(int size = lower; size <= upper; size++){
    if(verbose) cout << "running with <size x size> x <size x 1> " << size << endl;
    double sum = 0.0;
    const int m = size, n = 1, k = size;
    for(int rep = 0; rep <= reps; rep++){
      auto start = std::chrono::steady_clock::now();

      if constexpr (std::is_same_v<T, double>)
        stat = hipblasDgemmStridedBatched(GEMM_BATCHED_PARAMETERS);
      else if constexpr (std::is_same_v<T, float>)
        stat = hipblasSgemmStridedBatched(GEMM_BATCHED_PARAMETERS);
      else if constexpr (std::is_same_v<T, __half>)
        stat = hipblasHgemmStridedBatched(GEMM_BATCHED_PARAMETERS);

      hipDeviceSynchronize();
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      auto elapsed = time * 1e-3;

      if(stat != HIPBLAS_STATUS_SUCCESS){
        cerr << "Size " << size
             << " skipped because hipblas returned non-success status "
             << stat << endl;
        break;
      }

      if (rep != 0) sum += elapsed;
      
      if(verbose)
	cout << "size " << size << ": " << elapsed << " us; " 
	     << elapsed / num << " us per operation" << endl;
    }

    if (stat == HIPBLAS_STATUS_SUCCESS) {
      cout << "size " << size << " average execution time: " << sum / reps
           << " us; " << sum / reps / num << " us per operation; "
           << "floating-point operations per second: ";
      performance(m, n, k, 1e3 * (sum / reps / num));
    }

    // verify double precision operations 
    if constexpr (std::is_same_v<T, double>) {
      hipMemcpy(result, devResult, vectors_size, hipMemcpyDeviceToHost);
      gemmBatched_ref (num, upper, upper, 1, m, k, n, alpha, beta,
                       matrices, lda, vectors, ldb, result_ref, ldc);

      for (int i = 0; i < num; i++) {
      for (int j = 0; j < m; j++) {
        if (abs(result[i*upper+j] - result_ref[i*upper+j]) > 1e-6) {
          cout << "Mismatch at batch index " << i << ": " << result[i*upper+j] << "!="
               << result_ref[i*upper+j] << endl;
          break;
        }
      }}
    }
  }

  hipFree(devMatrices);
  hipFree(devVectors);
  hipFree(devResult);
  hipblasDestroy(handle);

  free(matrices);
  free(vectors);
  free(result);
  free(result_ref);
}

int main(int argc, char ** argv){

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
      num = strtoul(optarg, 0, 0);  // batch size
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
  gemmBatched<__half>(lower, upper, num, reps, verbose);
  cout << ">>>>>>>>>>>>>>> Single precision gemmBatched >>>>>>>>>>>>>>> " << endl;
  gemmBatched<float>(lower, upper, num, reps, verbose);
  cout << ">>>>>>>>>>>>>>> Double precision gemmBatched >>>>>>>>>>>>>>> " << endl;
  gemmBatched<double>(lower, upper, num, reps, verbose);
      
  return 0;
}
