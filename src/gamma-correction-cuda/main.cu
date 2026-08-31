//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include "utils.hpp"

__global__ 
void gamma_correction(ImgPixel* pixel, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // Lambda to process image with gamma = 2
  const float v = (0.3f * pixel[i].r + 0.59f * pixel[i].g + 0.11f * pixel[i].b) / 255.f;

  std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255.f * v * v);
  if (gamma_pixel > 255) gamma_pixel = 255;
  pixel[i].set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <image width> <image height> <block size> <repeat>\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int block_size = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  if (block_size <= 0) {
    fprintf(stderr, "Error: block size must be positive (got %d)\n", block_size);
    return 1;
  }

  // The average kernel time is divided by the repeat count.
  if (repeat <= 0) {
    fprintf(stderr, "Error: repeat count must be positive (got %d)\n", repeat);
    return 1;
  }

  // A BMP header stores the file size in 32-bit fields, so reject images
  // that cannot be represented before allocating anything.
  if (!ImgFormat::BMP::fits(width, height)) {
    fprintf(stderr,
            "Error: an image of %d x %d pixels cannot be represented as a "
            "BMP file (maximum size is %llu bytes)\n",
            width, height,
            static_cast<unsigned long long>(ImgFormat::BMP::maxByteSize));
    return 1;
  }

  Img<ImgFormat::BMP> image{width, height};
  ImgFractal fractal{width, height};

  // Lambda to process image with gamma = 2
  auto gamma_f = [](ImgPixel& pixel) {
    float v = (0.3f * pixel.r + 0.59f * pixel.g + 0.11f * pixel.b) / 255.f;

    std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255.f * v * v);
    if (gamma_pixel > 255) gamma_pixel = 255;
    pixel.set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
  };

  // fill image with created fractal
  int index = 0;
  image.fill([&index, width, &fractal](ImgPixel& pixel) {
      int x = index % width;
      int y = index / width;

      auto fractal_pixel = fractal(x, y);
      if (fractal_pixel < 0) fractal_pixel = 0;
      if (fractal_pixel > 255) fractal_pixel = 255;
      pixel.set(fractal_pixel, fractal_pixel, fractal_pixel, fractal_pixel);

      ++index;
  });

  Img<ImgFormat::BMP> image2 = image;
#ifdef DEBUG
  image.write("fractal_original.bmp");
#endif

  // call standard serial function for correctness check
  image.fill(gamma_f);
#ifdef DEBUG
  image.write("fractal_gamma_serial.bmp");
#endif

  ImgPixel* pixel;
  cudaMalloc((void**)&pixel, sizeof(ImgPixel) * image2.width() * image2.height());
  
  const int image_size = width * height;
  dim3 grids ((image_size + block_size - 1) / block_size);
  dim3 blocks (block_size); 

  cudaMemcpy(pixel, image2.data(), sizeof(ImgPixel) * image2.width() * image2.height(), cudaMemcpyHostToDevice);
  gamma_correction <<<grids, blocks>>> (pixel, image_size);
  cudaMemcpy(image2.data(), pixel, sizeof(ImgPixel) * image2.width() * image2.height(), cudaMemcpyDeviceToHost);

  // check correctness
  if (check(image.begin(), image.end(), image2.begin())) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    gamma_correction <<<grids, blocks>>> (pixel, image_size);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

#ifdef DEBUG
  image.write("fractal_gamma_parallel.bmp");
#endif
  cudaFree(pixel); 

  return 0;
}
