#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <hip/hip_runtime.h>
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

  float *d_image;
  hipMalloc((void**)&d_image, image_size);

  int blocks = std::min(PowTwoDivider(height), 64);
  dim3 dimBlockX (blocks);
  dim3 dimGridX ((height + blocks - 1) / blocks);

  blocks = std::min(PowTwoDivider(width), 64);
  dim3 dimBlockY (blocks);
  dim3 dimGridY ((width  + blocks - 1) / blocks);

  hipMemcpy(d_image, image, image_size, hipMemcpyHostToDevice);
  toCoef2DX<<<dimGridX, dimBlockX>>>(d_image, image_pitch, width, height);
  toCoef2DY<<<dimGridY, dimBlockY>>>(d_image, image_pitch, width, height);

  toCoef2DX_ref(image_ref, image_pitch, width, height);
  toCoef2DY_ref(image_ref, image_pitch, width, height);

  hipMemcpy(image, d_image, image_size, hipMemcpyDeviceToHost);

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

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    toCoef2DX<<<dimGridX, dimBlockX>>>(d_image, image_pitch, width, height);
    toCoef2DY<<<dimGridY, dimBlockY>>>(d_image, image_pitch, width, height);
  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", time * 1e-9f / repeat);

  hipFree(d_image);
  free(image);
  free(image_ref);
  if (!ok) return 1;
  return 0;
}
