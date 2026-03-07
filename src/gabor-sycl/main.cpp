#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

void gabor (
  sycl::nd_item<2> &item,
  double *gabor_spatial,
  const unsigned int height,
  const unsigned int width,
  const double center_y,
  const double center_x,
  const double ctheta,
  const double stheta,
  const double scale,
  const double sx_2,
  const double sy_2,
  const double fx)
{
  int x = item.get_global_id(1);
  int y = item.get_global_id(0);

  double centered_x, centered_y, u, v;

  if (x < width && y < height) {
    centered_y = (double)y - center_y;
    centered_x = (double)x - center_x;
    u = ctheta * centered_x - stheta * centered_y;
    v = ctheta * centered_y + stheta * centered_x;
    gabor_spatial[y*width + x] = scale * sycl::exp(-0.5*(u*u/sx_2 + v*v/sy_2)) *
                                 sycl::cos(2.0*M_PI*fx*u);
  }
}

double* generateGaborKernelDevice(
  const int repeat,
  const unsigned int height,
  const unsigned int width,
  const unsigned int par_T,
  const double par_L,
  const double theta)
{
  const double sx = (double)par_T / (2.0*sqrt(2.0*log(2.0)));
  const double sy = par_L * sx;
  const double sx_2 = sx*sx;
  const double sy_2 = sy*sy;
  const double fx = 1.0 / (double)par_T;
  const double ctheta = cos(theta);
  const double stheta = sin(theta);
  const double center_y = (double)height / 2.0;
  const double center_x = (double)width / 2.0;
  const double scale = 1.0/(2.0*M_PI*sx*sy);

  size_t image_size_bytes = height * width * sizeof(double);
  double *h_gabor_spatial = (double*) malloc (image_size_bytes);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  double *d_gabor_spatial = sycl::malloc_device<double>(height * width, q);
  q.memcpy(d_gabor_spatial, h_gabor_spatial, image_size_bytes);

  sycl::range<2> gws ((height+15)/16*16, (width+15)/16*16);
  sycl::range<2> lws (16, 16);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class image_process>(
         sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
         gabor(item,
               d_gabor_spatial,
               height,
               width, 
               center_y,
               center_x,
               ctheta,
               stheta,
               scale,
               sx_2,
               sy_2,
               fx);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(h_gabor_spatial, d_gabor_spatial, image_size_bytes).wait();
  sycl::free(d_gabor_spatial, q);

  return h_gabor_spatial;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <height> <width> <repeat>\n", argv[0]);
    return 1;
  }

  const int height = atoi(argv[1]);
  const int width = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const unsigned int par_T = 13;
  const double par_L = 2.65;
  const double theta = 45;

  double *h_filter = generateGaborKernelHost(height, width, par_T, par_L, theta);
  double *d_filter = generateGaborKernelDevice(repeat, height, width, par_T, par_L, theta);
  
  bool ok = true;
  for (int i = 0; i < width * height; i++) {
    if (fabs(h_filter[i] - d_filter[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
  free(h_filter);
  free(d_filter);
}
