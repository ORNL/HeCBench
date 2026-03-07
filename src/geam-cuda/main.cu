#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cublas_v2.h>

template <typename T>
void transpose_f64(int nrow, int ncol, int repeat) {
  int error = 0;
  const size_t size = nrow * ncol;
  const size_t size_byte = size * sizeof(T);

  T *matrix = (T *) malloc (size_byte);
  for (size_t i = 0; i < size; i++) matrix[i] = rand() % 13; 

  T *matrixT = (T *) malloc (size_byte);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < nrow; i++)
    for (int j = 0; j < ncol; j++)
      matrixT[(j*nrow)+i] = matrix[(i*ncol)+j];

  auto end = std::chrono::steady_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Host: serial matrix transpose time = %f (ms)\n", time * 1e-6f);

  cublasStatus_t stat;
  cublasHandle_t handle;
  cublasCreate(&handle);

  const T alpha = (T)1;
  const T beta  = (T)0;

  // store host and device results
  T *h_matrixT, *d_matrixT, *d_matrix;
  h_matrixT = (T *) malloc (size_byte);

  cudaMalloc((void**)&d_matrix, size_byte);
  cudaMemcpy(d_matrix, matrix, size_byte, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_matrixT, size_byte);

  time = 0.0;
  const int warmup = 4;

  for (int i = 0; i < repeat + warmup; i++) {
    if (i >= warmup) {
      start = std::chrono::steady_clock::now();
    }
    stat = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                       nrow, ncol,
                       &alpha, d_matrix, ncol,
                       &beta, d_matrix, nrow,
                       d_matrixT, nrow);
    cudaDeviceSynchronize(); // required for timing correctness
    if (i >= warmup) {
      end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    if (stat != CUBLAS_STATUS_SUCCESS) {
      error = 1;
      printf("Error: cublasDgeam failed to complete\n");
      break;
    }
  }

  printf("Device: average matrix transpose time = %f (ms)\n", (time * 1e-6f) / repeat);

  cudaMemcpy(h_matrixT, d_matrixT, size_byte, cudaMemcpyDeviceToHost);

  cudaFree(d_matrix);
  cudaFree(d_matrixT);

  if (error == 0) { // check host and device results
    error = memcmp(h_matrixT, matrixT, size_byte);
  }

  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  cublasDestroy(handle);
  free(h_matrixT);
  free(matrixT);
  free(matrix);
}

template <typename T>
void transpose_f32(int nrow, int ncol, int repeat) {
  int error = 0;
  const size_t size = nrow * ncol;
  const size_t size_byte = size * sizeof(T);

  T *matrix = (T *) malloc (size_byte);
  for (size_t i = 0; i < size; i++) matrix[i] = rand() % 13; 

  T *matrixT = (T *) malloc (size_byte);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < nrow; i++)
    for (int j = 0; j < ncol; j++)
      matrixT[(j*nrow)+i] = matrix[(i*ncol)+j];

  auto end = std::chrono::steady_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Host: serial matrix transpose time = %f (ms)\n", time * 1e-6f);

  cublasStatus_t stat;
  cublasHandle_t handle;
  cublasCreate(&handle);

  const T alpha = (T)1;
  const T beta  = (T)0;

  // store host and device results
  T *h_matrixT, *d_matrixT, *d_matrix;
  h_matrixT = (T *) malloc (size_byte);
  cudaMalloc((void**)&d_matrix, size_byte);
  cudaMemcpy(d_matrix, matrix, size_byte, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_matrixT, size_byte);

  time = 0.0;
  const int warmup = 4;

  for (int i = 0; i < repeat + warmup; i++) {
    if (i >= warmup) {
      start = std::chrono::steady_clock::now();
    }
    stat = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                       nrow, ncol,
                       &alpha, d_matrix, ncol,
                       &beta, d_matrix, nrow,
                       d_matrixT, nrow);
    cudaDeviceSynchronize(); // required for timing correctness
    if (i >= warmup) {
      end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    if (stat != CUBLAS_STATUS_SUCCESS) {
      error = 1;
      printf("Error: cublasSgeam failed to complete\n");
      break;
    }
  }

  printf("Device: average matrix transpose time = %f (ms)\n", (time * 1e-6f) / repeat);

  cudaMemcpy(h_matrixT, d_matrixT, size_byte, cudaMemcpyDeviceToHost);

  cudaFree(d_matrix);
  cudaFree(d_matrixT);

  if (error == 0) { // check host and device results
    error = memcmp(h_matrixT, matrixT, size_byte);
  }

  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  cublasDestroy(handle);
  free(matrixT);
  free(matrix);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage %s <matrix row> <matrix col> <repeat>\n", argv[0]);
    return 1;
  }

  int nrow = atoi(argv[1]); 
  int ncol = atoi(argv[2]); 
  const int repeat = atoi(argv[3]); 
  if (nrow <= 0 || ncol <= 0 || repeat < 0) {
    printf("Error: invalid inputs\n");
    return 1;
  }

  printf("----------------FP32 transpose matrix (%d x %d)----------------\n",
         nrow, ncol);
  transpose_f32<float>(nrow, ncol, repeat);

  printf("----------------FP64 transpose matrix (%d x %d)----------------\n",
         nrow, ncol);
  transpose_f64<double>(nrow, ncol, repeat);

  return 0;
}
