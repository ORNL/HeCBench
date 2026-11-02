#ifndef REFERENCE_H
#define REFERENCE_H

#include <cstdio>
#include <cstdlib>

// sad_array holds one entry per template placement that lies fully inside
// the image, stored row-major over the (image - template + 1) placements.
bool reference(const int *sad_array,
               const unsigned char *image,
               const unsigned char *kernel,
               const int sad_array_size,
               const int image_width,
               const int image_height,
               const int kernel_width,
               const int kernel_height,
               const int threshold,
               const int min_mse,
               const int num_occurrences)
{
  const int out_width = image_width - kernel_width + 1;
  const int out_height = image_height - kernel_height + 1;
  const int kernel_size = kernel_width * kernel_height;

  if (out_width * out_height != sad_array_size) {
    printf("SAD array size %d does not match %d x %d placements\n",
           sad_array_size, out_width, out_height);
    return false;
  }

  int *ref = new int[sad_array_size];
  int ref_min = threshold;

  for (int row = 0; row < out_height; row++) {
    for (int col = 0; col < out_width; col++) {
      int sad_result = 0;
      for (int kr = 0; kr < kernel_height; kr++) {
        for (int kc = 0; kc < kernel_width; kc++) {
          const int image_addr = ((row + kr) * image_width + (col + kc)) * 3;
          const int kernel_addr = (kr * kernel_width + kc) * 3;
          sad_result += abs((int)image[image_addr + 0] - (int)kernel[kernel_addr + 0]) +
                        abs((int)image[image_addr + 1] - (int)kernel[kernel_addr + 1]) +
                        abs((int)image[image_addr + 2] - (int)kernel[kernel_addr + 2]);
        }
      }
      const int norm_sad = (int)(sad_result / (float)kernel_size);
      ref[row * out_width + col] = norm_sad;
      if (norm_sad < ref_min) ref_min = norm_sad;
    }
  }

  int ref_occurrences = 0;
  for (int i = 0; i < sad_array_size; i++)
    if (ref[i] == ref_min) ref_occurrences++;

  bool ok = true;
  int mismatches = 0;
  for (int i = 0; i < sad_array_size; i++) {
    if (sad_array[i] != ref[i]) {
      if (mismatches < 10)
        printf("SAD mismatch at placement (row %d, col %d): %d != %d (expected)\n",
               i / out_width, i % out_width, sad_array[i], ref[i]);
      mismatches++;
    }
  }
  if (mismatches) {
    printf("%d of %d SAD entries differ from the reference\n",
           mismatches, sad_array_size);
    ok = false;
  }

  if (min_mse != ref_min) {
    printf("Minimum mismatch: %d != %d (expected)\n", min_mse, ref_min);
    ok = false;
  }
  if (num_occurrences != ref_occurrences) {
    printf("Occurrence mismatch: %d != %d (expected)\n",
           num_occurrences, ref_occurrences);
    ok = false;
  }

  delete[] ref;
  return ok;
}

#endif
