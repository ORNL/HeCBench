// *********************************************************************
// A simple demo application that implements a
// vector dot product computation between two arrays.
//
// Runs computations with on the GPU device and then checks results
// *********************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

template <typename T>
void dot (const size_t iNumElements, const int iNumIterations)
{
  const size_t src_size = iNumElements;
  const size_t src_size_bytes = src_size * sizeof(T);

  // Allocate and initialize host arrays
  T* srcA = (T*) malloc (src_size_bytes);
  T* srcB = (T*) malloc (src_size_bytes);
  T  dst;

  size_t i;
  double sum = 0.0;
  for (i = 0; i < iNumElements ; ++i)
  {
    srcA[i] = (T)(sqrt(65504.0 / iNumElements));
    srcB[i] = (T)(sqrt(65504.0 / iNumElements));
    sum += (float)srcA[i] * (float)srcB[i];
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  T *d_srcA = sycl::malloc_device<T>(src_size, q);
  q.memcpy(d_srcA, srcA, src_size_bytes);

  T *d_srcB = sycl::malloc_device<T>(src_size, q);
  q.memcpy(d_srcB, srcB, src_size_bytes);

  T *d_dst = sycl::malloc_device<T>(1, q);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iNumIterations; i++) {
    oneapi::mkl::blas::dot(q, iNumElements, d_srcA, 1, d_srcB, 1, d_dst);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average oneMKL::dot execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);
  q.memcpy(&dst, d_dst, sizeof(T)).wait();
  printf("Host: %lf  Device: %lf\n", sum, double(dst));
  printf("%s\n\n", (fabs(double(dst) - sum) < 1e-1) ? "PASS" : "FAIL");

  sycl::free(d_dst, q);
  sycl::free(d_srcA, q);
  sycl::free(d_srcB, q);

  free(srcA);
  free(srcB);
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t iNumElements = atol(argv[1]);
  const int iNumIterations = atoi(argv[2]);

  printf("\nFP64 Dot\n");
  dot<double>(iNumElements, iNumIterations);
  printf("\nFP32 Dot\n");
  dot<float>(iNumElements, iNumIterations);
  printf("\nFP16 Dot\n");
  dot<sycl::half>(iNumElements, iNumIterations);
  printf("\nBF16 Dot\n");
  dot<sycl::ext::oneapi::bfloat16>(iNumElements, iNumIterations);

  if (!(fabs(double(dst) - sum) < 1e-1)) return 1;
  return EXIT_SUCCESS;
}
