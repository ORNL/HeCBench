#include <iostream>
#include <chrono>
#include <omp.h>
#include "bitmap_image.hpp"

#define BLOCK_SIZE_X  16
#define BLOCK_SIZE_Y  16
#define BLOCK_SIZE    (BLOCK_SIZE_X * BLOCK_SIZE_Y)

#define THRESHOLD     20
#define min(a, b) ((a) < (b) ? (a) : (b))

void compute_sad_array(
                    int*__restrict sad_array,
    const unsigned char*__restrict image,
    const unsigned char*__restrict kernel,
    int sad_array_size,
    int& min_mse,
    int& num_occurrences,
    int image_width, int image_height,
    int kernel_width, int kernel_height,
    int kernel_size,
    double &kernel_time)
{
  auto kbegin = std::chrono::steady_clock::now();

  #pragma omp target teams distribute parallel for collapse(2) thread_limit(BLOCK_SIZE)
  for (int row = 0; row <= image_height - kernel_height; row++) {
    for (int col = 0; col <= image_width - kernel_width; col++) {
      int sad_result = 0;
      const int overlap_width = min(image_width - col, kernel_width);
      const int overlap_height = min(image_height - row, kernel_height);
      #pragma unroll 4
      for (int kr = 0; kr < overlap_height; kr++) {
        #pragma unroll 4
        for (int kc = 0; kc < overlap_width; kc++) {
          const int image_addr = ((row + kr) * image_width + (col + kc)) * 3;
          const int kernel_addr = (kr * kernel_width + kc) * 3;
          const int m_r = (int)(image[image_addr + 0]);
          const int m_g = (int)(image[image_addr + 1]);
          const int m_b = (int)(image[image_addr + 2]);
          const int t_r = (int)(kernel[kernel_addr + 0]);
          const int t_g = (int)(kernel[kernel_addr + 1]);
          const int t_b = (int)(kernel[kernel_addr + 2]);
          const int error = abs(m_r - t_r) + abs(m_g - t_g) + abs(m_b - t_b);
          sad_result += error;
        }
      }

      int norm_sad = (int)(sad_result / (float)kernel_size);

      int my_index_in_sad_array = row * (image_width - kernel_width + 1) + col;
      if (my_index_in_sad_array < sad_array_size) {
        sad_array[my_index_in_sad_array] = norm_sad;
      }
    }
  }

  int m = THRESHOLD;
  #pragma omp target teams distribute parallel for thread_limit(256) \
    map(tofrom: m) reduction(min: m)
  for (int i = 0; i < sad_array_size; i++) 
    m = min(m, sad_array[i]);

  int n = 0; 
  #pragma omp target teams distribute parallel for thread_limit(256) \
    map(tofrom: n) reduction(+: n)
  for (int i = 0; i < sad_array_size; i++) {
    if (sad_array[i] == m) n++;
  }

  auto kend = std::chrono::steady_clock::now();
  kernel_time += std::chrono::duration_cast<std::chrono::milliseconds> (kend - kbegin).count();

  min_mse = m;
  num_occurrences = n;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: ./main <image> <template image> <repeat>\n";
    return 1;
  }

  bitmap_image main_image(argv[1]);
  bitmap_image template_image(argv[2]);
  const int repeat = atoi(argv[3]);

  const int main_width = main_image.width();
  const int main_height = main_image.height();
  const int main_size = main_width * main_height;

  const int template_width = template_image.width();
  const int template_height = template_image.height();
  const int template_size = template_width * template_height;

  const int height_difference = main_height - template_height;
  const int width_difference = main_width - template_width;
  const int sad_array_size = (height_difference + 1) * (width_difference + 1);

  // Host allocation
  unsigned char* h_main_image = new unsigned char[3 * main_size];

  for (int row = 0; row < main_height; row++) {
    for (int col = 0; col < main_width; col++) {
      rgb_t colors;
      main_image.get_pixel(col, row, colors);
      h_main_image[(row * main_width + col) * 3 + 0] = colors.red;
      h_main_image[(row * main_width + col) * 3 + 1] = colors.green;
      h_main_image[(row * main_width + col) * 3 + 2] = colors.blue;
    }
  }

  unsigned char* h_template_image = new unsigned char[3 * template_size];

  for (int row = 0; row < template_height; row++) {
    for (int col = 0; col < template_width; col++) {
      rgb_t colors;
      template_image.get_pixel(col, row, colors);
      h_template_image[(row * template_width + col) * 3 + 0] = colors.red;
      h_template_image[(row * template_width + col) * 3 + 1] = colors.green;
      h_template_image[(row * template_width + col) * 3 + 2] = colors.blue;
    }
  }

  int* h_sad_array = new int[sad_array_size];
  int h_num_occurances;
  int h_min_mse;
  float elapsed_time; 

  #pragma omp target data map(to: h_main_image[0:3*main_size],\
                                  h_template_image[0:3*template_size]) \
                          map(alloc: h_sad_array[0:sad_array_size])
  {
    // Measure device execution time
    double kernel_time = 0.0;

    auto begin = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {

      compute_sad_array(
          h_sad_array, h_main_image, h_template_image, sad_array_size, 
          h_min_mse, h_num_occurances,
          main_width, main_height,
          template_width, template_height, template_size,
          kernel_time);
    }

    auto end = std::chrono::steady_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();

    std::cout << "Parallel Computation Results: " << std::endl;
    std::cout << "Kernel time in msec: " << kernel_time << std::endl; 
    std::cout << "Elapsed time in msec = " << elapsed_time << std::endl; 
    std::cout << "Main Image Dimensions: " << main_width << "*" << main_height << std::endl;
    std::cout << "Template Image Dimensions: " << template_width << "*" << template_height << std::endl;
    std::cout << "Found Minimum:  " << h_min_mse << std::endl;
    std::cout << "Number of Occurances: " << h_num_occurances << std::endl;
  }

  delete[] h_main_image;
  delete[] h_template_image;
  delete[] h_sad_array;
  return 0;
}
