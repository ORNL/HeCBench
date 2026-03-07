#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

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
  double *gabor_spatial = (double*) malloc (image_size_bytes);

  #pragma omp target data map (from: gabor_spatial[0:height * width])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          double centered_y = (double)y - center_y;
          double centered_x = (double)x - center_x;
          double u = ctheta * centered_x - stheta * centered_y;
          double v = ctheta * centered_y + stheta * centered_x;
          gabor_spatial[y*width + x] = scale * exp(-0.5*(u*u/sx_2 + v*v/sy_2)) * cos(2.0*M_PI*fx*u);
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);
  }

  return gabor_spatial;
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
