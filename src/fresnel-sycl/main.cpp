#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <sycl/sycl.hpp>

SYCL_EXTERNAL
double Fresnel_Sine_Integral(double);

void reference (const double *__restrict input,
                      double *__restrict output, const int n) {
  for (int i = 0; i < n; i++)
    output[i] = Fresnel_Sine_Integral(input[i]);
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // range [0, 8], interval 1e-7
  const double interval = 1e-7;
  const int points = (int)(8.0 / interval);
  const size_t points_size = points * sizeof(double);
  double *x = (double*) malloc (points_size);
  double *output = (double*) malloc (points_size);
  double *h_output = (double*) malloc (points_size);
  for (int i = 0; i < points; i++)
    x[i] = (double)i * interval;
	   
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  double *d_x = sycl::malloc_device<double>(points, q);
  q.memcpy(d_x, x, points_size);

  double *d_output = sycl::malloc_device<double>(points, q);

  sycl::range<1> gws ((points + 255)/256*256);
  sycl::range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class fresnel>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < points) d_output[i] = Fresnel_Sine_Integral(d_x[i]);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(output, d_output, points_size).wait();

  // verify
  reference(x, h_output, points);
  bool ok = true;
  for (int i = 0; i < points; i++) {
    if (fabs(h_output[i] - output[i]) > 1e-6) {
      printf("%lf %lf\n", h_output[i], output[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  
  sycl::free(d_x, q);
  sycl::free(d_output, q);
  free(x);
  free(output);
  free(h_output);
  if (!ok) return 1;
  return 0;
}
