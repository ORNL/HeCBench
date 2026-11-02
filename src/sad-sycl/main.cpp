#include <iostream>
#include <chrono>
#include <sycl/sycl.hpp>
#include "bitmap_image.hpp"

#define BLOCK_SIZE_X  16
#define BLOCK_SIZE_Y  16
#define BLOCK_SIZE    (BLOCK_SIZE_X * BLOCK_SIZE_Y)

#define THRESHOLD     20
#define FOUND_MIN     5000
#define min(a, b) ((a) < (b) ? (a) : (b))
#define syncthreads() item.barrier(sycl::access::fence_space::local_space)

#include "reference.h"

void compute_sad_array(
    sycl::nd_item<2> &item,
                    int*__restrict sad_array,
    const unsigned char*__restrict image,
    const unsigned char*__restrict kernel,
    const int sad_array_size,
    const int image_width,
    const int image_height,
    const int kernel_width,
    const int kernel_height,
    const int kernel_size)
{
  int col = item.get_global_id(1);
  int row = item.get_global_id(0);
  int sad_result = 0;

  if (row <= image_height - kernel_height && col <= image_width - kernel_width) {
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
        const int error = sycl::abs(m_r - t_r) + sycl::abs(m_g - t_g) + sycl::abs(m_b - t_b);
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


void find_min_in_sad_array(
    sycl::nd_item<1> &item,
    const int sad_array_size,
          int* __restrict cache,
    const int* __restrict sad_array,
          int* __restrict min_sad)
{
  unsigned int lid = item.get_local_id(0);
  unsigned int bsz = item.get_local_range(0);
  unsigned int gid = item.get_group(0) * bsz + lid;
  unsigned int stride = item.get_group_range(0) * bsz;
  unsigned int offset = 0;

  int temp = FOUND_MIN;
  while (gid + offset < sad_array_size) {
    temp = min(temp, sad_array[gid + offset]);
    offset += stride;
  }

  cache[lid] = temp;

  syncthreads();

  unsigned int i = bsz / 2;
  while (i != 0) {
    if (lid < i)
      cache[lid] = min(cache[lid], cache[lid + i]);
    syncthreads();
    i /= 2;
  }

  // Update global min for each block
  if (lid == 0) {
    //atomicMin(min_sad, cache[0]);
    auto ao = sycl::atomic_ref<int,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space> (min_sad[0]);
    ao.fetch_min(cache[0]);
  }
}

void get_num_of_occurrences(
    sycl::nd_item<1> &item,
    const int sad_array_size,
          int &s,
    const int*__restrict sad_array,
    const int*__restrict min_sad,
          int*__restrict num_occurrences)
{
  unsigned int gid = item.get_global_id(0);

  if (gid < sad_array_size) {
    unsigned int lid = item.get_local_id(0);

    if (lid == 0) s = 0;

    syncthreads();

    if (sad_array[gid] == *min_sad) {
      // atomicAdd(&cache[0], 1);
      auto ao = sycl::atomic_ref<int,
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space> (s);
      ao.fetch_add(1);
    }

    syncthreads();

    // Update global occurance for each block
    if (lid == 0) {
      // atomicAdd(num_occurrences, cache[0]);
      auto ao = sycl::atomic_ref<int,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device,
              sycl::access::address_space::global_space> (num_occurrences[0]);
      ao.fetch_add(s);
    }
  }
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Device allocation
  unsigned char *d_main_image = sycl::malloc_device<unsigned char>(3 * main_size, q);
  q.memcpy(d_main_image, h_main_image, 3 * main_size * sizeof(unsigned char));

  unsigned char *d_template_image = sycl::malloc_device<unsigned char>(3 * template_size, q);
  q.memcpy(d_template_image, h_template_image, 3 * template_size * sizeof(unsigned char));

  int *d_sad_array = sycl::malloc_device<int>(sad_array_size, q);
  int *d_min_mse = sycl::malloc_device<int>(1, q);
  int *d_num_occurances = sycl::malloc_device<int>(1, q);
  q.memset(d_sad_array, 0xFF, sad_array_size * sizeof(int));

  sycl::range<2> gws ((unsigned int)ceilf((float)main_height / BLOCK_SIZE_Y) * BLOCK_SIZE_Y,
                      (unsigned int)ceilf((float)main_width / BLOCK_SIZE_X) * BLOCK_SIZE_X );
  sycl::range<2> lws (BLOCK_SIZE_Y, BLOCK_SIZE_X);

  sycl::range<1> gws2 ((unsigned int)ceilf((float)sad_array_size / BLOCK_SIZE) * BLOCK_SIZE);
  sycl::range<1> lws2 (BLOCK_SIZE);

  // Measure device execution time
  double kernel_time = 0.0;

  auto begin = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {

    h_min_mse = THRESHOLD;

    q.memset(d_num_occurances, 0, sizeof(int));
    q.memcpy(d_min_mse, &h_min_mse, sizeof(int));

    q.wait();
    auto kbegin = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class sad>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        compute_sad_array (
          item,
          d_sad_array,
          d_main_image,
          d_template_image,
          sad_array_size,
          main_width, main_height,
          template_width, template_height,
          template_size);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<int> cache (lws2, cgh);
      cgh.parallel_for<class find_min>(
        sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
        find_min_in_sad_array (
          item,
          sad_array_size,
          cache.get_multi_ptr<sycl::access::decorated::no>().get(),
          d_sad_array,
          d_min_mse);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<int, 0> sum (cgh);
      cgh.parallel_for<class count>(
        sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
        get_num_of_occurrences (
          item,
          sad_array_size,
          sum,
          d_sad_array,
          d_min_mse,
          d_num_occurances);
      });
    });

    q.wait();
    auto kend = std::chrono::steady_clock::now();
    kernel_time += std::chrono::duration_cast<std::chrono::milliseconds> (kend - kbegin).count();

    q.memcpy(&h_min_mse, d_min_mse, sizeof(int));
    q.memcpy(&h_num_occurances, d_num_occurances, sizeof(int));
  }
  q.wait();

  auto end = std::chrono::steady_clock::now();
  float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();

  q.memcpy(h_sad_array, d_sad_array, sad_array_size * sizeof(int)).wait();

  std::cout << "Parallel Computation Results: " << std::endl;
  std::cout << "Kernel time in msec: " << kernel_time << std::endl;
  std::cout << "Elapsed time in msec = " << elapsed_time << std::endl;
  std::cout << "Main Image Dimensions: " << main_width << "*" << main_height << std::endl;
  std::cout << "Template Image Dimensions: " << template_width << "*" << template_height << std::endl;
  std::cout << "Found Minimum:  " << h_min_mse << std::endl;
  std::cout << "Number of Occurances: " << h_num_occurances << std::endl;

  bool ok = reference(h_sad_array, h_main_image, h_template_image, sad_array_size,
                      main_width, main_height, template_width, template_height,
                      THRESHOLD, h_min_mse, h_num_occurances);
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  sycl::free(d_main_image, q);
  sycl::free(d_template_image, q);
  sycl::free(d_sad_array, q);
  sycl::free(d_min_mse, q);
  sycl::free(d_num_occurances, q);
  delete[] h_main_image;
  delete[] h_template_image;
  delete[] h_sad_array;
  return ok ? 0 : 1;
}
