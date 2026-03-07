#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main (int argc, char* argv[]){
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  // repeat at least once
  const int repeat = max(1, atoi(argv[1]));

  bool ok = true;
  cudaError_t cudaStat;
  cublasStatus_t cublasStat;
  cublasHandle_t handle;

  cublasStat = cublasCreate(&handle);
  if (cublasStat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
  }

  // store the nrm2 results
  float* result = (float*) malloc (repeat * sizeof(float));
  if (!result) {
    printf ("result memory allocation failed");
    return 1;
  }

  float *a = NULL;
  float *d_a = NULL;

  for (int n = 512*1024; n <= 1024*1024*512; n = n * 2) {
    int i, j;
    size_t size = n * sizeof(float);
    a = (float *) malloc (size);
    if (!a) {
      printf ("host memory allocation failed");
      if (d_a) {
        cudaStat = cudaFree(d_a);
        if (cudaStat != cudaSuccess)
          printf ("device memory allocation failed");
      }
      break;
    }

    // reference
    double gold = 0.0;  // double is required to match host and device results 
    for (i = 0; i < n; i++) {
      a[i] = (float)((i+1) % 7);
      gold += a[i]*a[i];
    }
    gold = sqrt(gold);

    cudaStat = cudaMalloc((void**)&d_a, size);
    if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      if (a) free(a);
      break;
    }

    cudaStat = cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) {
      printf ("device memory copy failed");
    }

    //-------------------------------------------------------------------
    // handle |        | input | handle to the cuBLAS library context.
    //      n |        | input | number of elements in the vector x.
    //      x | device | input |<type> vector with n elements.
    //   incx |        | input | stride between consecutive elements of x.
    // result | host or device | output | the resulting norm, which is 0.0 if n,incx<=0.
    //-------------------------------------------------------------------
    auto kstart = std::chrono::steady_clock::now();
    
    for (j = 0; j < repeat; j++) {
      cublasStat = cublasSnrm2(handle, n, d_a, 1, result+j);
      if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS Snrm2 failed\n");
      }
    }

    auto kend = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
    printf("#elements = %.2f M: average cublasSnrm2 execution time = %f (us), performance = %f (Gop/s)\n",
           n / (1024.f*1024.f), (ktime * 1e-3f) / repeat, 1.f * (2*n+1) * repeat / ktime);

    cudaStat = cudaFree(d_a);
    if (cudaStat != cudaSuccess) {
      printf ("device memory deallocation failed");
    }

    // snrm2 results match across all iterations
    for (j = 0; j < repeat; j++) 
      if (fabsf((float)gold - result[j]) > 1e-1f) {
        printf("FAIL at iteration %d: gold=%f actual=%f for %d elements\n",
               j, (float)gold, result[j], i);
        ok = false;
        break;
      }

    free(a);
  }

  free(result);
  cublasStat = cublasDestroy(handle);
  if (cublasStat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS destroy failed\n");
  }

  if (ok) printf("PASS\n");
  if (!ok) exit(1);
  return 0;
}
