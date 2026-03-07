#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <sycl/sycl.hpp>
#include "kernels.h"
#include "reference.h"

int PowTwoDivider(int n)
{
  if (n == 0) return 0;
  int divider = 1;
  while ((n & divider) == 0) divider <<= 1;
  return divider;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int image_pitch = width * sizeof(float);
  const int numPix = width * height;
  const int image_size = numPix * sizeof(float);

  float *image = (float*) malloc (image_size);
  float *image_ref = (float*) malloc (image_size);

  // image with random values
  std::mt19937 gen{ 123 };
  std::normal_distribution<float> d{0.f, 1.f};

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      image_ref[h * width + w] = image[h * width + w] = d(gen);
    }
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_image = sycl::malloc_device<float>(numPix, q);

  int blocks = std::min(PowTwoDivider(height), 64);
  sycl::range<1> lwsX (blocks);
  sycl::range<1> gwsX ((height + blocks-1) / blocks * blocks);

  blocks = std::min(PowTwoDivider(width), 64);
  sycl::range<1> lwsY (blocks);
  sycl::range<1> gwsY ((width + blocks-1) / blocks * blocks);

  q.memcpy(d_image, image, image_size);
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(gwsX, lwsX), [=] (sycl::nd_item<1> item) {
      toCoef2DX(item, d_image, image_pitch, width, height);
    });
  });

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(gwsY, lwsY), [=] (sycl::nd_item<1> item) {
      toCoef2DY(item, d_image, image_pitch, width, height);
    });
  });

  toCoef2DX_ref(image_ref, image_pitch, width, height);
  toCoef2DY_ref(image_ref, image_pitch, width, height);
  q.memcpy(image, d_image, image_size).wait();

  bool ok = true;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      if (std::fabs(image_ref[h * width + w] - image[h * width + w]) > 1e-3f) {
        ok = false;
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<1>(gwsX, lwsX), [=] (sycl::nd_item<1> item) {
        toCoef2DX(item, d_image, image_pitch, width, height);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<1>(gwsY, lwsY), [=] (sycl::nd_item<1> item) {
        toCoef2DY(item, d_image, image_pitch, width, height);
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", time * 1e-9f / repeat);

  sycl::free(d_image, q);
  free(image);
  free(image_ref);
  return 0;
}
