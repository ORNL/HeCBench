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
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

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

  T *d_srcA;
  T *d_srcB;
  T *d_dst;

  cudaMalloc((void**)&d_srcA, src_size_bytes);
  cudaMemcpy(d_srcA, srcA, src_size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_srcB, src_size_bytes);
  cudaMemcpy(d_srcB, srcB, src_size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_dst, sizeof(T));


  cublasHandle_t h;
  cublasCreate(&h);
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);

  cudaDataType xType, yType, rType, eType;
  if constexpr (std::is_same<T, double>::value) {
    xType = yType = rType = eType = CUDA_R_64F;
  } else if constexpr (std::is_same<T, float>::value) {
    xType = yType = rType = eType = CUDA_R_32F;
  } else if constexpr (std::is_same<T, __half>::value) {
    xType = yType = rType = CUDA_R_16F;
    eType = CUDA_R_32F;
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    xType = yType = rType = CUDA_R_16BF;
    eType = CUDA_R_32F;
  }

  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    cublasDotEx(h, iNumElements, d_srcA, xType, 1, d_srcB,
                yType, 1, d_dst, rType, eType);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average cublasDotEx execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);

  cudaMemcpy(&dst, d_dst, sizeof(T), cudaMemcpyDeviceToHost);
  printf("Host: %lf  Device: %lf\n", sum, double(dst));
  printf("%s\n\n", (fabs(double(dst) - sum) < 1e-1) ? "PASS" : "FAIL");

  cudaFree(d_dst);
  cudaFree(d_srcA);
  cudaFree(d_srcB);
  cublasDestroy(h);

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
  dot<__half>(iNumElements, iNumIterations);
  printf("\nBF16 Dot\n");
  dot<__nv_bfloat16>(iNumElements, iNumIterations);

  if (!(fabs(double(dst) - sum) < 1e-1)) return 1;
  return EXIT_SUCCESS;
}
