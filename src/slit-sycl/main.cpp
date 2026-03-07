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
#include <complex>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"
#include "reference.h"

// Compute 2D FFT with oneMKL

double fft2(std::complex<double> *inData, std::complex<double> *outData, const unsigned int N,
            const int repeat)
{
  double time = 0.0;
  try {
    // Catch asynchronous exceptions
    auto exception_handler = [] (sycl::exception_list exceptions) {
      for (std::exception_ptr const& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch(sycl::exception const& e) {
          std::cout << "Caught asynchronous SYCL exception:" << std::endl
                    << e.what() << std::endl;
        }
      }
    };

    sycl::queue q(
#ifdef USE_GPU
      sycl::gpu_selector_v,
#else
      sycl::cpu_selector_v,
#endif
      exception_handler, sycl::property::queue::in_order());

    const size_t data_bytes = sizeof(std::complex<double>) * N * N;
    std::complex<double> *d_inData =
       (std::complex<double> *)sycl::malloc_device(data_bytes, q);

    // create descriptors
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                 oneapi::mkl::dft::domain::COMPLEX> desc({N, N});

    // variadic set_value
    desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                   static_cast<std::int64_t>(1));
    desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                   oneapi::mkl::dft::config_value::INPLACE); // DFTI_INPLACE

    // commit_descriptor
    desc.commit(q);

    for (int i = 0; i < repeat; i++) {
      q.memcpy(d_inData, inData, data_bytes).wait();

      auto start = std::chrono::steady_clock::now();

      // compute_forward
      oneapi::mkl::dft::compute_forward(desc, d_inData).wait();

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    q.memcpy(outData, d_inData, data_bytes).wait();

    sycl::free(d_inData, q);
  }
  catch(sycl::exception const& e) {
      std::cout << "\t\tSYCL exception during FFT" << std::endl;
      std::cout << "\t\t" << e.what() << std::endl;
  }
  catch(std::runtime_error const& e) {
      std::cout << "\t\truntime exception during FFT" << std::endl;
      std::cout << "\t\t" << e.what() << std::endl;
  }

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
  std::complex<double> *inputData =
      (std::complex<double> *)malloc(N * N * sizeof(std::complex<double>));

  // FFT result
  std::complex<double> *fftData =
      (std::complex<double> *)malloc(N * N * sizeof(std::complex<double>));

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
      inputData[j * N + i] = std::complex<double>(0.0, 0.0);
      inputData_ref[j * N + i] = 0.0;
      if ((abs(i-N/2) <= slit_dist+slit_width) &&
          (abs(i-N/2) >= slit_dist) &&
          (abs(j-N/2) <= slit_height)){
        inputData[j * N + i] = std::complex<double>(1.0, 0.0);
        inputData_ref[j * N + i] = 1.0;
      }
    }
  }

  double avg_time = fft2(inputData, fftData, N, repeat);

  printf("Average execution time of FFT: %lf ms\n", avg_time);

  for(int i = 0; i < N*N; i++){
    outputData[i] = fftData[i].real() * fftData[i].real() +
                    fftData[i].imag() * fftData[i].imag();
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
