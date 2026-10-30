#include <iostream>
#include <chrono>
#include <hip/hip_runtime.h>
#include "bitmap_image.hpp"

#define check(stmt)                                          \
  do {                        \
    hipError_t err = stmt;   \
    if (err != hipSuccess) { \
      printf("[ERROR] Failed to run stmt %d, error body: %s\n", __LINE__, hipGetErrorString(err));  \
      return -1; }            \
  } while (0)                 \

#define BLOCK_SIZE_X  16
#define BLOCK_SIZE_Y  16
#define BLOCK_SIZE    (BLOCK_SIZE_X * BLOCK_SIZE_Y)

#define THRESHOLD     20
#define FOUND_MIN     5000
#define min(a, b) ((a) < (b) ? (a) : (b))

__global__ void compute_sad_array(
                    int*__restrict__ sad_array,
    const unsigned char*__restrict__ image,
    const unsigned char*__restrict__ kernel,
    const int sad_array_size,
    const int image_width,
    const int image_height,
    const int kernel_width,
    const int kernel_height,
    const int kernel_size)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
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

__global__ void find_min_in_sad_array(
    const int sad_array_size,
    const int* __restrict__ sad_array,
          int* __restrict__ min_sad)
{
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = gridDim.x * blockDim.x;
  unsigned int offset = 0;

  __shared__ int cache[BLOCK_SIZE];

  int temp = FOUND_MIN;
  while (gid + offset < sad_array_size) {
    temp = min(temp, sad_array[gid + offset]);
    offset += stride;
  }

  cache[threadIdx.x] = temp;

  __syncthreads();

  unsigned int i = blockDim.x / 2;
  while (i != 0) {
    if (threadIdx.x < i)
      cache[threadIdx.x] = min(cache[threadIdx.x], cache[threadIdx.x + i]);
    __syncthreads();
    i /= 2;
  }

  // Update global min for each block
  if (threadIdx.x == 0)
    atomicMin(min_sad, cache[0]);
}

__global__ void get_num_of_occurrences(
    const int sad_array_size,
    const int*__restrict__ sad_array,
    const int*__restrict__ min_sad,
          int*__restrict__ num_occurrences)
{
  unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ int s;

  if (gid < sad_array_size) {

    if (threadIdx.x == 0) s = 0;

    __syncthreads();

    if (sad_array[gid] == *min_sad)
      atomicAdd(&s, 1);

    __syncthreads();

    // Update global occurance for each block
    if (threadIdx.x == 0)
      atomicAdd(num_occurrences, s);
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

  // Device allocation
  unsigned char* d_main_image;
  unsigned char* d_template_image;
  int* d_sad_array;
  int* d_min_mse;
  int* d_num_occurances;

  check(hipMalloc((void **)&d_main_image, 3 * main_size * sizeof(unsigned char)));
  check(hipMalloc((void **)&d_template_image, 3 * template_size * sizeof(unsigned char)));
  check(hipMalloc((void **)&d_sad_array, sad_array_size * sizeof(int)));
  check(hipMalloc((void **)&d_min_mse, sizeof(int)));
  check(hipMalloc((void **)&d_num_occurances, sizeof(int)));

  dim3 grids((unsigned int)ceilf((float)main_width / BLOCK_SIZE_X),
             (unsigned int)ceilf((float)main_height / BLOCK_SIZE_Y));
  dim3 blocks(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

  dim3 grids_2((unsigned int)ceilf((float)sad_array_size / BLOCK_SIZE));
  dim3 blocks_2(BLOCK_SIZE);

  check(hipMemcpy(d_main_image, h_main_image,
                   3 * main_size * sizeof(unsigned char), hipMemcpyHostToDevice));
  check(hipMemcpy(d_template_image, h_template_image,
                   3 * template_size * sizeof(unsigned char), hipMemcpyHostToDevice));

  // Measure device execution time
  double kernel_time = 0.0;

  auto begin = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {

    h_min_mse = THRESHOLD;
    check(hipMemset(d_num_occurances, 0, sizeof(int)));
    check(hipMemcpy(d_min_mse, &h_min_mse, sizeof(int), hipMemcpyHostToDevice));

    hipDeviceSynchronize();
    auto kbegin = std::chrono::steady_clock::now();

    hipLaunchKernelGGL(compute_sad_array, grids, blocks , 0, 0, 
        d_sad_array, d_main_image, d_template_image, sad_array_size, 
        main_width, main_height, template_width, template_height, template_size);

    hipLaunchKernelGGL(find_min_in_sad_array, grids_2, blocks_2 , 0, 0, 
        sad_array_size, d_sad_array, d_min_mse);

    hipLaunchKernelGGL(get_num_of_occurrences, grids_2, blocks_2 , 0, 0, 
        sad_array_size, d_sad_array, d_min_mse, d_num_occurances);

    hipDeviceSynchronize();
    auto kend = std::chrono::steady_clock::now();
    kernel_time += std::chrono::duration_cast<std::chrono::milliseconds> (kend - kbegin).count();

    check(hipMemcpy(&h_min_mse, d_min_mse, sizeof(int), hipMemcpyDeviceToHost));
    check(hipMemcpy(&h_num_occurances, d_num_occurances, sizeof(int), hipMemcpyDeviceToHost));
  }

  auto end = std::chrono::steady_clock::now();
  float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();

  std::cout << "Parallel Computation Results: " << std::endl;
  std::cout << "Kernel time in msec: " << kernel_time << std::endl; 
  std::cout << "Elapsed time in msec: " << elapsed_time << std::endl; 
  std::cout << "Main Image Dimensions: " << main_width << "*" << main_height << std::endl;
  std::cout << "Template Image Dimensions: " << template_width << "*" << template_height << std::endl;
  std::cout << "Found Minimum: " << h_min_mse << std::endl;
  std::cout << "Number of Occurances: " << h_num_occurances << std::endl;

  check(hipFree(d_main_image));
  check(hipFree(d_template_image));
  check(hipFree(d_sad_array));
  check(hipFree(d_min_mse));
  check(hipFree(d_num_occurances));
  delete[] h_main_image;
  delete[] h_template_image;
  delete[] h_sad_array;
  return 0;
}
