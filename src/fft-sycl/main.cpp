#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>

#ifdef SINGLE_PRECISION
#define T float 
#define T2 sycl::float2
#define EPISON 1e-4
#else
#define T double
#define T2 sycl::double2
#define EPISON 1e-6
#endif

void submit_fft_base(sycl::queue& q, T2* work, size_t globalsz, size_t localsz);
void submit_ifft_base(sycl::queue& q, T2* work, size_t globalsz, size_t localsz);
void submit_fft_optimized(sycl::queue& q, T2* work, size_t globalsz, size_t localsz);
void submit_ifft_optimized(sycl::queue& q, T2* work, size_t globalsz, size_t localsz);

#include "reference.h"

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <problem size> <number of passes>\n", argv[0]);
    printf("Problem size [0-3]: 0=1M, 1=8M, 2=96M, 3=256M\n");
    return 1;
  }

  srand(2);
  int i;
  int select = atoi(argv[1]);
  int passes = atoi(argv[2]);

  // Convert to MB
  int probSizes[4] = { 1, 8, 96, 256 };
  unsigned long bytes = probSizes[select];
  bytes *= 1024 * 1024;

  // now determine how much available memory will be used
  const int half_n_ffts = bytes / (512*sizeof(T2)*2);
  const int n_ffts = half_n_ffts * 2;
  const int half_n_cmplx = half_n_ffts * 512;
  const int n_cmplx = half_n_cmplx*2;
  unsigned long used_bytes = n_cmplx * sizeof(T2);

  fprintf(stdout, "used_bytes=%lu, n_cmplx=%d\n", used_bytes, n_cmplx);

  // allocate host memory, in-place FFT/iFFT operations
  T2 *source = (T2*) malloc (used_bytes);
  T2 *reference = (T2*) malloc (used_bytes);

  auto init_host_data = [&]() {
    for (i = 0; i < half_n_cmplx; i++) {
      source[i].x() = sinf(i / powf(10000, i % 768 / 384));
      source[i].y() = cosf(i / powf(10000, i % 768 / 384));
      source[i+half_n_cmplx].x() = source[i].x();
      source[i+half_n_cmplx].y()= source[i].y();
    }
    memcpy(reference, source, used_bytes);
  };

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  T2 *work = sycl::malloc_device<T2>(n_cmplx, q);
  q.memcpy(work, source, used_bytes);

  const size_t localsz = 64;
  const size_t globalsz = localsz * n_ffts;

  auto verify_fft = [&]() {
    for (int i = 0; i < n_cmplx; i++) {
      if ( std::fabs((T)source[i].x() - (T)reference[i].x()) > EPISON) {
        return true;
      }
      if ( std::fabs((T)source[i].y() - (T)reference[i].y()) > EPISON) {
        return true;
      }
    }
    return false;
  };

  auto verify_ifft = [&]() {
    for (int i = 0; i < n_cmplx; i++) {
      int j = i % half_n_cmplx;
      if (fabs((T)source[i].x() - (T)sinf(j / powf(10000, j%768/384))) > EPISON) {
        return true;
      }
      if (fabs((T)source[i].y() - (T)cosf(j / powf(10000, j%768/384))) > EPISON) {
        return true;
      }
    }
    return false;
  };

  auto run_base = [&]() {
    std::cout << "=== Base kernel ===" << std::endl;
    init_host_data();
    q.memcpy(work, source, used_bytes).wait();

    submit_fft_base(q, work, globalsz, localsz);
    fft1D_512_reference<64>(reference, n_ffts);
    q.memcpy(source, work, used_bytes).wait();
    std::cout << "FFT " << (verify_fft() ? "FAIL" : "PASS")  << std::endl;

    submit_ifft_base(q, work, globalsz, localsz);
    q.memcpy(source, work, used_bytes).wait();
    std::cout << "iFFT " << (verify_ifft() ? "FAIL" : "PASS")  << std::endl;

    auto start = std::chrono::steady_clock::now();
    for (int k=0; k<passes; k++) {
      submit_fft_base(q, work, globalsz, localsz);
      submit_ifft_base(q, work, globalsz, localsz);
    }
    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / passes << " (s)\n";
  };

  auto run_optimized = [&]() {
    std::cout << "=== Optimized kernel ===" << std::endl;
    init_host_data();
    q.memcpy(work, source, used_bytes).wait();

    submit_fft_optimized(q, work, globalsz, localsz);
    fft1D_512_reference<64>(reference, n_ffts);
    q.memcpy(source, work, used_bytes).wait();
    std::cout << "FFT " << (verify_fft() ? "FAIL" : "PASS")  << std::endl;

    submit_ifft_optimized(q, work, globalsz, localsz);
    q.memcpy(source, work, used_bytes).wait();
    std::cout << "iFFT " << (verify_ifft() ? "FAIL" : "PASS")  << std::endl;

    auto start = std::chrono::steady_clock::now();
    for (int k=0; k<passes; k++) {
      submit_fft_optimized(q, work, globalsz, localsz);
      submit_ifft_optimized(q, work, globalsz, localsz);
    }
    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / passes << " (s)\n";
  };

  run_base();
  run_optimized();

  sycl::free(work, q);

  free(reference);
  free(source);

  return 0;
}
