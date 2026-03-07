#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  // repeat at least once
  const int repeat = std::max(1, atoi(argv[1]));

  bool ok = true;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // store the nrm2 results
  float* h_result = (float*) sycl::malloc_host (repeat * sizeof(float), q);
  if (h_result == nullptr) {
    printf ("output on host allocation failed");
    return 1;
  }

  float* d_result = (float*) sycl::malloc_device (repeat * sizeof(float), q);
  if (d_result == nullptr) {
    printf ("output on device allocation failed");
    return 1;
  }

  // store the mkl nrm2 status
  std::vector<sycl::event> status (repeat);

  float *a = nullptr;
  float *d_a = nullptr;

  for (int n = 512*1024; n <= 1024*1024*512; n = n * 2) {
    int i, j;
    size_t size = n * sizeof(float);
    a = (float*) malloc (size);
    if (a == nullptr) {
      printf ("input on host allocation failed");
      if (d_a != nullptr) sycl::free(d_a, q);
      break;
    }

    // reference
    double gold = 0.0;  // double is required to match host and device results
    for (i = 0; i < n; i++) {
      a[i] = (float)((i+1) % 7);
      gold += a[i]*a[i];
    }
    gold = sqrt(gold);

    d_a = (float *)sycl::malloc_device(size, q);
    if (d_a == nullptr) {
      printf ("input on device allocation failed");
      if (a != nullptr) free(a);
      break;
    }

    q.memcpy(d_a, a, size).wait();

    auto kstart = std::chrono::steady_clock::now();

    try {
      for (j = 0; j < repeat; j++) {
        status[j] = oneapi::mkl::blas::column_major::nrm2(q, n, d_a, 1, d_result+j);
        q.memcpy(h_result + j, d_result + j, sizeof(float), status[j]);
      }
    } catch(sycl::exception const& e) {
      std::cout << "\t\tCaught synchronous SYCL exception during NRM2:\n"
                << e.what() << std::endl;
    }

    q.wait();
    auto kend = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
    printf("#elements = %.2f M: average mkl::blas::column_major::nrm2 execution time = %f (us), performance = %f (Gop/s)\n",
           n / (1024.f*1024.f), (ktime * 1e-3f) / repeat, 1.f * (2*n+1) * repeat / ktime);

    sycl::free(d_a, q);

    // snrm2 results match across all iterations
    for (j = 0; j < repeat; j++)
      if (fabsf((float)gold - h_result[j]) > 1e-1f) {
        printf("FAIL at iteration %d: gold=%f actual=%f for %d elements\n",
               j, (float)gold, h_result[j], i);
        ok = false;
        break;
      }

    free(a);
  }

  sycl::free(h_result, q);
  sycl::free(d_result, q);

  if (ok) printf("PASS\n");
  if (!ok) exit(1);
  return 0;
}
