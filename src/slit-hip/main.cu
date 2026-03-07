/***********************************************************
 *
 * Developed for Seminar in Parallelisation of Physics
 * Calculations on GPUs with CUDA, Department of Physics
 * Technical University of Munich.
 *
 * Author: Binu Amaratunga
 *
 *
 ***********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <hip/hip_complex.h>
#include "reference.h"

// Compute 2D FFT with hipFFT

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
  if (code != hipSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
  }
}

double fft2(hipDoubleComplex *inData, hipDoubleComplex *outData,
          const unsigned int N, const int repeat)
{
  hipfftDoubleComplex *d_inData = NULL;

  const size_t data_bytes = sizeof(hipfftDoubleComplex) * N * N;

  gpuErrChk(hipMalloc(&d_inData, data_bytes));

  hipfftResult flag;
  hipfftHandle plan;

  flag = hipfftPlan2d(&plan, N, N, HIPFFT_Z2Z);
  if ( HIPFFT_SUCCESS != flag ) printf("2D: hipfftPlan2d fails!\n");

  double time = 0.0;
  for (int i = 0; i < repeat; i++) {
    gpuErrChk(hipMemcpy(d_inData, inData, data_bytes, hipMemcpyHostToDevice));

    gpuErrChk(hipDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();

    flag = hipfftExecZ2Z(plan, d_inData, d_inData, HIPFFT_FORWARD);
    if ( HIPFFT_SUCCESS != flag ) printf("2D: hipfftExecR2C fails!\n");

    gpuErrChk(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  gpuErrChk(hipMemcpy(outData, d_inData, data_bytes, hipMemcpyDeviceToHost));

  flag = hipfftDestroy(plan);
  if ( HIPFFT_SUCCESS != flag ) printf("2D: hipfftDestroy fails!\n");
  gpuErrChk(hipFree(d_inData));

  return time * 1e-6 / repeat;
}


int main(int argc, char** argv){

  if (argc != 3) {
    printf("Usage: %s <the transform size in the x and y dimensions> <repeat>\n",
           argv[0]);
    return 1;
  }
  const int N = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  printf("Running FFT for %d x %d = %d = 2 ^ %d data points...\n",
         N, N, N*N, (int)(log(N*N)/log(2)));

  // Complex data input
  hipDoubleComplex * inputData = (hipDoubleComplex *)malloc(N * N * sizeof(hipDoubleComplex));

  // FFT result
  hipDoubleComplex * fftData = (hipDoubleComplex *)malloc(N * N * sizeof(hipDoubleComplex));

  // Real data
  double * outputData = (double *)malloc(N * N * sizeof(double));

  // reference output
  double * inputData_ref = (double *)malloc(N * N * sizeof(double));
  double * outputData_ref = (double *)malloc(N * N * sizeof(double));

  const int slit_height = 4;
  const int slit_width  = 2;
  const int slit_dist   = 8;

  // Create double slit
  for (int j = 0; j < N; j++){
    for (int i = 0; i < N; i++){
      inputData[j * N + i] = make_hipDoubleComplex(0.0, 0.0);
      inputData_ref[j * N + i] = 0.0;
      if ((abs(i-N/2) <= slit_dist+slit_width) &&
          (abs(i-N/2) >= slit_dist) &&
          (abs(j-N/2) <= slit_height)){
        inputData[j * N + i] = make_hipDoubleComplex(1.0, 0.0);
        inputData_ref[j * N + i] = 1.0;
      }
    }
  }

  double avg_time = fft2(inputData, fftData, N, repeat);

  printf("Average execution time of FFT: %lf ms\n", avg_time);

  for(int i = 0; i < N*N; i++){
    outputData[i] = hipCreal(fftData[i]) * hipCreal(fftData[i]) +
                    hipCimag(fftData[i]) * hipCimag(fftData[i]);
  }

  reference(inputData_ref, outputData_ref, N);

  bool ok = true;
  for (int i = 0; i < N * N; i++) {
    if (fabs(outputData[i] - outputData_ref[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  free(inputData);
  free(inputData_ref);
  free(fftData);
  free(outputData);
  free(outputData_ref);

  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  return 0;
}
