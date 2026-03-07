/*
  Reference
  Chapter 7 in Programming massively parallel processors,
  A hands-on approach (D. Kirk and W. Hwu)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#define MAX_MASK_WIDTH 10
#define MAX_BLOCK_SIZE 1024

template<typename T, int BLOCK_SIZE>
void conv1d(const T * __restrict__ mask,
            const T * __restrict__ in,
                  T * __restrict__ out,
            const int input_width,
            const int mask_width)
{
  #pragma omp target teams distribute parallel for num_threads(BLOCK_SIZE)
  for (int i = 0; i < input_width; i++) {
    int start = i - mask_width / 2;
    T s = 0;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        s += in[start + j] * mask[j];
      }
    }
    out[i] = s;
  }
}

template<typename T, int BLOCK_SIZE>
void conv1d_tiled(const T *__restrict__ mask,
                  const T *__restrict__ in,
                        T *__restrict__ out,
                  const int input_width,
                  const int mask_width)
{
  #pragma omp target teams num_teams(input_width/BLOCK_SIZE)
  {
    T tile[BLOCK_SIZE + MAX_MASK_WIDTH - 1];
    #pragma omp parallel num_threads(BLOCK_SIZE)
    {
      int bid = omp_get_team_num();
      int lid = omp_get_thread_num();
      int dim = omp_get_num_threads();
      int i = bid * dim + lid;

      int n = mask_width / 2;  // last n cells of the previous tile

      // load left cells 
      int halo_left = (bid - 1) * dim + lid;
      if (lid >= dim - n)
         tile[lid - (dim - n)] = halo_left < 0 ? 0 : in[halo_left];

      // load center cells
      tile[n + lid] = in[bid * dim + lid];

      // load right cells
      int halo_right = (bid + 1) * dim + lid;
      if (lid < n)
         tile[lid + dim + n] = halo_right >= input_width ? 0 : in[halo_right];

      #pragma omp barrier

      T s = 0;
      for (int j = 0; j < mask_width; j++)
        s += tile[lid + j] * mask[j];

      out[i] = s;
    }
  }
}

template<typename T, int BLOCK_SIZE>
void conv1d_tiled_caching(const T *__restrict__ mask,
                          const T *__restrict__ in,
                                T *__restrict__ out,
                          const int input_width,
                          const int mask_width)
{
  #pragma omp target teams num_teams(input_width/BLOCK_SIZE)
  {
    T tile[BLOCK_SIZE];
    #pragma omp parallel num_threads(BLOCK_SIZE)
    {
      int bid = omp_get_team_num();
      int lid = omp_get_thread_num();
      int dim = omp_get_num_threads();
      int i = bid * dim + lid;
      tile[lid] = in[i];
      #pragma omp barrier

      int this_tile_start = bid * dim;
      int next_tile_start = (bid + 1) * dim;
      int start = i - (mask_width / 2);
      T s = 0;
      for (int j = 0; j < mask_width; j++) {
        int in_index = start + j;
        if (in_index >= 0 && in_index < input_width) {
          if (in_index >= this_tile_start && in_index < next_tile_start) {
            // in_index = (start + j) = (i - mask_width/2 +j) >= 0,
            // then map in_index to tile_index
            s += tile[lid + j - (mask_width / 2)] * mask[j];
          } else {
            s += in[in_index] * mask[j];
          }
        }
      }
      out[i] = s;
    }
  }
}

template <typename T>
void reference(const T *h_in,
               const T *d_out,
               const T *mask,
               const int input_width,
               const int mask_width)
{
  bool ok = true;
  for (int i = 0; i < input_width; i++) {
    T s = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        s += h_in[start + j] * mask[j];
      }
    }
    if (fabs(s - d_out[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
}

template <typename T>
void conv1D(const int input_width, const int mask_width, const int repeat)
{
  size_t size_bytes = input_width * sizeof(T);

  T *a, *b;
  a = (T *)malloc(size_bytes); // input
  b = (T *)malloc(size_bytes); // output

  T mask[MAX_MASK_WIDTH];

  for (int i = 0; i < MAX_MASK_WIDTH; i++) mask[i] = 1; 

  srand(123);
  for (int i = 0; i < input_width; i++) {
    a[i] = rand() % 256;
  }

  #pragma omp target data map(to: a[0:input_width], \
                                  mask[0:mask_width]) \
                          map(alloc: b[0:input_width])
  {
    // conv1D basic
    for (int bs = 64; bs <= MAX_BLOCK_SIZE; bs = bs * 2) {
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++) {
        switch(bs) {
          case   64: {conv1d<T, 64>(mask, a, b, input_width, mask_width); break;}
          case  128: {conv1d<T, 128>(mask, a, b, input_width, mask_width); break;}
          case  256: {conv1d<T, 256>(mask, a, b, input_width, mask_width); break;}
          case  512: {conv1d<T, 512>(mask, a, b, input_width, mask_width); break;}
          case 1024: {conv1d<T, 1024>(mask, a, b, input_width, mask_width); break;}
        }
      }
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time of conv1d kernel: %f (us)\n",
             (time * 1e-3f) / repeat);
      #pragma omp target update from (b[0:input_width])
      reference(a, b, mask, input_width, mask_width);
    }

    // conv1D tiling
    for (int bs = 64; bs <= MAX_BLOCK_SIZE; bs = bs * 2) {
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++) {
        switch(bs) {
          case   64: {conv1d_tiled<T, 64>(mask, a, b, input_width, mask_width); break;}
          case  128: {conv1d_tiled<T, 128>(mask, a, b, input_width, mask_width); break;}
          case  256: {conv1d_tiled<T, 256>(mask, a, b, input_width, mask_width); break;}
          case  512: {conv1d_tiled<T, 512>(mask, a, b, input_width, mask_width); break;}
          case 1024: {conv1d_tiled<T, 1024>(mask, a, b, input_width, mask_width); break;}
        }
      }
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time of conv1d-tiled kernel: %f (us)\n",
             (time * 1e-3f) / repeat);
      #pragma omp target update from (b[0:input_width])
      reference(a, b, mask, input_width, mask_width);
    }

    // conv1D tiling and caching
    for (int bs = 64; bs <= MAX_BLOCK_SIZE; bs = bs * 2) {
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; i++) {
        switch(bs) {
          case   64: {conv1d_tiled_caching<T, 64>(mask, a, b, input_width, mask_width); break;}
          case  128: {conv1d_tiled_caching<T, 128>(mask, a, b, input_width, mask_width); break;}
          case  256: {conv1d_tiled_caching<T, 256>(mask, a, b, input_width, mask_width); break;}
          case  512: {conv1d_tiled_caching<T, 512>(mask, a, b, input_width, mask_width); break;}
          case 1024: {conv1d_tiled_caching<T, 1024>(mask, a, b, input_width, mask_width); break;}
        }
      }
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time of conv1d-tiled-caching kernel: %f (us)\n",
             (time * 1e-3f) / repeat);
      #pragma omp target update from (b[0:input_width])
      reference(a, b, mask, input_width, mask_width);
    }
  }

  free(a);
  free(b);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <input_width> <repeat>\n", argv[0]);
    return 1;
  }

  int input_width = atoi(argv[1]);
  // a multiple of BLOCK_SIZE
  input_width = (input_width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE * MAX_BLOCK_SIZE;

  const int repeat = atoi(argv[2]);

  for (int mask_width = 3; mask_width < MAX_MASK_WIDTH; mask_width += 2) {
    printf("\n---------------------\n");
    printf("Mask width: %d\n", mask_width); 

    printf("1D convolution (FP64)\n");
    conv1D<double>(input_width, mask_width, repeat);

    printf("1D convolution (FP32)\n");
    conv1D<float>(input_width, mask_width, repeat);

    printf("1D convolution (INT16)\n");
    conv1D<int16_t>(input_width, mask_width, repeat);
  }

  return 0;
}
