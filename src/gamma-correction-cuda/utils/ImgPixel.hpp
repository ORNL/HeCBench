//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef _GAMMA_UTILS_IMGPIXEL_HPP
#define _GAMMA_UTILS_IMGPIXEL_HPP

#include <cstdint>
#include <ostream>

// The CUDA and HIP variants compile this header with a device compiler and
// need the accessors annotated; the OpenMP and SYCL variants do not.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define HECBENCH_HOST_DEVICE __host__ __device__
#else
#define HECBENCH_HOST_DEVICE
#endif

// struct to store a pixel of image
struct ImgPixel {
  std::uint8_t b;
  std::uint8_t g;
  std::uint8_t r;
  std::uint8_t a;

  bool operator==(ImgPixel const& other) const {
    return (b == other.b) && (g == other.g) && (r == other.r) && (a == other.a);
  }

  bool operator!=(ImgPixel const& other) const { return !(*this == other); }

  HECBENCH_HOST_DEVICE
  void set(std::uint8_t blue, std::uint8_t green, std::uint8_t red,
           std::uint8_t alpha) {
    b = blue;
    g = green;
    r = red;
    a = alpha;
  }
};

std::ostream& operator<<(std::ostream& output, ImgPixel const& pixel) {
  return output << "(" << unsigned(pixel.r) << ", " << unsigned(pixel.g) << ", "
                << unsigned(pixel.b) << ", " << unsigned(pixel.a) << ")";
}

#endif  // _GAMMA_UTILS_IMGPIXEL_HPP
