#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include "util.h"
#include "image.h"
#include "kernel.h"

// 16-byte zero used by the device memset. One store covers four pixels.
struct alignas(16) uint4 { unsigned int x, y, z, w; };

int main(int argc, char* argv[]) 
{
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int input_image_pitch = width;
  const int output_image_pitch = width * 4;
  const int numPix = width * height;
  const int input_image_size = numPix * sizeof(uchar);
  const int output_image_size = numPix * 4 * sizeof(uchar);

  uchar *input = (uchar*) malloc (input_image_size);
  uchar *output = (uchar*) malloc (output_image_size);

  // RGGB, GRBG, GBRG, BGGR
  const int bayer_pattern = RGGB;

  // input image with random values
  srand(123);
  for (int i = 0; i < numPix; i++) {
    input[i] = rand() % 256;
  }

  const uint teamX = (width + tile_cols - 1) / tile_cols;
  const uint teamY = (height + tile_rows - 1) / tile_rows;

  #pragma omp target data map(to: input[0:numPix]) map(from: output[0:4*numPix])
  {
    auto start = std::chrono::steady_clock::now();

    // Device memset: 16-byte stores, plus a short tail when numPix is not a multiple of 4.
    uint4 *out4 = reinterpret_cast<uint4 *>(output);
    unsigned int *out32 = reinterpret_cast<unsigned int *>(output);
    const int nvec = numPix / 4;
    const int tail = numPix % 4;

    //this version takes a tile (z=1) and each tile job does 4 line median sorts
    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams distribute parallel for
      for (int p = 0; p < nvec; p++)
        out4[p] = uint4{0, 0, 0, 0};
      if (tail) {
        #pragma omp target teams distribute parallel for
        for (int p = nvec * 4; p < numPix; p++)
          out32[p] = 0;
      }
      malvar_he_cutler_demosaic (
        teamX, teamY,
        height, width, 
        input, input_image_pitch,
        output, output_image_pitch,
        bayer_pattern );
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", time * 1e-9f / repeat);
  }

  long sum = 0;
  for (int i = 0; i < numPix; i++) sum += output[i];
  printf("Checksum: %ld\n", sum);

  free(input);
  free(output);
  return 0;
}
