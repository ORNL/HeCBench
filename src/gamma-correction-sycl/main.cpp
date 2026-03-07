//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iomanip>
#include <iostream>
#include <chrono>
#include <sycl/sycl.hpp>
#include "utils.hpp"
#include <cstdlib>

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <image width> <image height> <block size> <repeat>\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int block_size = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  ImgPixel *pixel = sycl::malloc_device<ImgPixel>(
    image2.width() * image2.height(), q);

  sycl::range<1> gws (width * height);
  sycl::range<1> lws (block_size); 

  float total_time = 0.f;

  for (int i = 0; i < repeat; i++) {
    q.memcpy(pixel, image2.data(), sizeof(ImgPixel) * image2.width() * image2.height()).wait();

    auto start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler& cgh) { 
      cgh.parallel_for<class gamma_correction>( 
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0); 
        // Lambda to process image with gamma = 2
        const float v = (0.3f * pixel[i].r + 0.59f * pixel[i].g + 0.11f * pixel[i].b) / 255.f;
        std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255.f * v * v);
        if (gamma_pixel > 255) gamma_pixel = 255;
        pixel[i].set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }
  
  printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / repeat);

  q.memcpy(image2.data(), pixel, sizeof(ImgPixel) * image2.width() * image2.height()).wait();

  // check correctness
  if (check(image.begin(), image.end(), image2.begin())) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
    exit(1);
  }

#ifdef DEBUG
  image.write("fractal_gamma_parallel.bmp");
#endif
  sycl::free(pixel, q); 

  return 0;
}
