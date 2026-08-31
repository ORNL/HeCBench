//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef _GAMMA_UTILS_IMGFORMAT_HPP
#define _GAMMA_UTILS_IMGFORMAT_HPP

#include "ImgPixel.hpp"

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace ImgFormat {

// struct to store an image in BMP format
struct BMP {
 private:
  using FileHeader = struct {
    // not from specification
    // was added for alignemt
    // store size of rest of the fields
    std::uint16_t sizeRest;  // file header size in bytes

    std::uint16_t type;
    std::uint32_t size;  // file size in bytes
    std::uint32_t reserved;
    std::uint32_t offBits;  // cumulative header size in bytes
  };

  using InfoHeader = struct {
    // from specification
    // store size of rest of the fields
    std::uint32_t size;  // info header size in bytes

    std::int32_t width;   // image width in pixels
    std::int32_t height;  // image height in pixels
    std::uint16_t planes;
    std::uint16_t bitCount;      // color depth
    std::uint32_t compression;   // compression
    std::uint32_t sizeImage;     // image map size in bytes
    std::int32_t xPelsPerMeter;  // pixel per metre (y axis)
    std::int32_t yPelsPerMeter;  // pixel per metre (y axis)
    std::uint32_t clrUsed;       // color pallete (0 is default)
    std::uint32_t clrImportant;
  };

  FileHeader _fileHeader;
  InfoHeader _infoHeader;

  // Cumulative size of the file header and the info header, in bytes.
  static constexpr std::uint64_t headerSize = 14 + 40;

  // Total size in bytes an image of the given dimensions would occupy.
  // Requires width > 0 and height > 0. Cannot overflow: the largest possible
  // value, for a 2147483647 x 2147483647 image, is about 1.8e19, which still
  // fits in a std::uint64_t (max ~1.84e19).
  static std::uint64_t byteSize(std::int32_t width,
                                std::int32_t height) noexcept {
    const std::uint64_t w = static_cast<std::uint64_t>(width);
    const std::uint64_t h = static_cast<std::uint64_t>(height);

    const std::uint64_t padSize = (4 - (w * sizeof(ImgPixel)) % 4) % 4;
    return w * h * sizeof(ImgPixel) + h * padSize + headerSize;
  }

 public:
  // Largest file size representable in the 32-bit BMP header fields.
  static constexpr std::uint64_t maxByteSize =
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());

  // Non-throwing query: true when an image of the given dimensions can be
  // described by a BMP header. Callers that prefer to fail gracefully can
  // consult this before constructing; the constructor itself throws.
  static bool fits(std::int32_t width, std::int32_t height) noexcept {
    if (width <= 0 || height <= 0) return false;
    return byteSize(width, height) <= maxByteSize;
  }

  // Throws std::invalid_argument for non-positive dimensions and
  // std::length_error when the image is too large to describe in a BMP
  // header. Use fits() to test in advance.
  BMP(std::int32_t width, std::int32_t height) { reset(width, height); }

  void reset(std::int32_t width, std::int32_t height) {
    // The BMP header stores the image map size and the total file size in
    // unsigned 32-bit fields. Validate in 64-bit arithmetic rather than
    // silently wrapping around and emitting a corrupt header.
    if (width <= 0 || height <= 0) {
      throw std::invalid_argument(
          "BMP: image dimensions must be positive (got " +
          std::to_string(width) + " x " + std::to_string(height) + ")");
    }

    const std::uint64_t allSize64 = byteSize(width, height);
    if (allSize64 > maxByteSize) {
      throw std::length_error(
          "BMP: an image of " + std::to_string(width) + " x " +
          std::to_string(height) + " pixels requires " +
          std::to_string(allSize64) +
          " bytes, which exceeds the maximum BMP file size of " +
          std::to_string(maxByteSize) +
          " bytes representable in the 32-bit header fields");
    }

    const std::uint32_t mapSize =
        static_cast<std::uint32_t>(allSize64 - headerSize);
    const std::uint32_t allSize = static_cast<std::uint32_t>(allSize64);

    _fileHeader.sizeRest = 14;  // file header size in bytes
    _fileHeader.type = 0x4d42;
    _fileHeader.size = allSize;  // file size in bytes
    _fileHeader.reserved = 0;
    _fileHeader.offBits = 54;  // sizeRest + size -> 14 + 40 -> 54

    _infoHeader.size = 40;        // info header size in bytes
    _infoHeader.width = width;    // image width in pixels
    _infoHeader.height = height;  // image height in pixels
    _infoHeader.planes = 1;
    _infoHeader.bitCount = 32;        // color depth
    _infoHeader.compression = 0;      // compression
    _infoHeader.sizeImage = mapSize;  // image map size in bytes
    _infoHeader.xPelsPerMeter = 0;    // pixel per metre (x axis)
    _infoHeader.yPelsPerMeter = 0;    // pixel per metre (y axis)
    _infoHeader.clrUsed = 0;          // color pallete (0 is default)
    _infoHeader.clrImportant = 0;
  }

  template <template <class> class Image, typename Format>
  void write(std::ofstream& ostream, Image<Format> const& image) const {
    ostream.write(reinterpret_cast<char const*>(&_fileHeader.type),
                  _fileHeader.sizeRest);

    ostream.write(reinterpret_cast<char const*>(&_infoHeader),
                  _infoHeader.size);

    ostream.write(reinterpret_cast<char const*>(image.data()),
                  static_cast<std::streamsize>(image.width()) *
                      image.height() * sizeof(image.data()[0]));
  }

  FileHeader const& fileHeader() const noexcept { return _fileHeader; }
  InfoHeader const& infoHeader() const noexcept { return _infoHeader; }
};

}  // namespace ImgFormat

#endif  // _GAMMA_UTILS_IMGFORMAT_HPP
