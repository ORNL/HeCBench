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
#include <sycl/sycl.hpp>

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif

#define MAX_MASK_WIDTH 10
#define MAX_BLOCK_SIZE 1024

template<typename T>
void conv1d(sycl::nd_item<1> &item,
            const T * __restrict__ mask,
            const T * __restrict__ in,
                  T * __restrict__ out,
            const int input_width,
            const int mask_width)
{
  int i = item.get_global_id(0);
  T s = 0;
  int start = i - mask_width / 2;
  for (int j = 0; j < mask_width; j++) {
    if (start + j >= 0 && start + j < input_width) {
      s += in[start + j] * ldg(&mask[j]);
    }
  }
  out[i] = s;
}

template<typename T>
void conv1d_tiled(sycl::nd_item<1> &item,
                        T *__restrict__ tile,
                  const T * __restrict__ mask,
                  const T *__restrict__ in,
                        T *__restrict__ out,
                  const int input_width,
                  const int mask_width)
{
  int lid = item.get_local_id(0);
  int bid = item.get_group(0);
  int dim = item.get_local_range(0);
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

  item.barrier(sycl::access::fence_space::local_space);

  T s = 0;
  for (int j = 0; j < mask_width; j++)
    s += tile[lid + j] * ldg(&mask[j]);

  out[i] = s;
}

template<typename T>
void conv1d_tiled_caching(sycl::nd_item<1> &item,
                                T *__restrict__ tile,
                          const T *__restrict__ mask,
                          const T *__restrict__ in,
                                T *__restrict__ out,
                          const int input_width,
                          const int mask_width)
{
  int lid = item.get_local_id(0);
  int bid = item.get_group(0);
  int dim = item.get_local_range(0);
  int i = bid * dim + lid;
  tile[lid] = in[i];

  item.barrier(sycl::access::fence_space::local_space);

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
        s += tile[lid + j - (mask_width / 2)] * ldg(&mask[j]);
      } else {
        s += in[in_index] * ldg(&mask[j]);
      }
    }
  }
  out[i] = s;
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
void conv1D(sycl::queue &q, const int input_width, const int mask_width, const int repeat)
{
  size_t size_bytes = input_width * sizeof(T);

  T *a, *b;
  a = (T *)malloc(size_bytes); // input
  b = (T *)malloc(size_bytes); // output

  T h_mask[MAX_MASK_WIDTH];

  for (int i = 0; i < MAX_MASK_WIDTH; i++) h_mask[i] = 1; 

  srand(123);
  for (int i = 0; i < input_width; i++) {
    a[i] = rand() % 256;
  }

  T *mask, *d_a, *d_b;
  mask = sycl::malloc_device<T>(MAX_MASK_WIDTH, q);
  d_a = sycl::malloc_device<T>(input_width, q);
  d_b = sycl::malloc_device<T>(input_width, q);

  q.memcpy(d_a, a, size_bytes);
  q.memcpy(mask, h_mask, mask_width * sizeof(T));

  q.wait();

  // conv1D basic
  for (int bs = 64; bs <= MAX_BLOCK_SIZE; bs = bs * 2) {
    sycl::range<1> gws (input_width);
    sycl::range<1> lws (bs);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          conv1d(item, mask, d_a, d_b, input_width, mask_width);
        });
      });
    }
    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    q.memcpy(b, d_b, size_bytes).wait();
    reference(a, b, h_mask, input_width, mask_width);
  }

  // conv1D tiling
  for (int bs = 64; bs <= MAX_BLOCK_SIZE; bs = bs * 2) {
    sycl::range<1> gws (input_width);
    sycl::range<1> lws (bs);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<T, 1> tile (sycl::range<1>(bs + MAX_MASK_WIDTH - 1), cgh);
        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          conv1d_tiled(item,
                       tile.template get_multi_ptr<sycl::access::decorated::no>().get(),
                       mask, d_a, d_b, input_width, mask_width);
        });
      });
    }
    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d-tiled kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    q.memcpy(b, d_b, size_bytes).wait();
    reference(a, b, h_mask, input_width, mask_width);
  }

  // conv1D tiling and caching
  for (int bs = 64; bs <= MAX_BLOCK_SIZE; bs = bs * 2) {
    sycl::range<1> gws (input_width);
    sycl::range<1> lws (bs);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<T, 1> tile (sycl::range<1>(bs), cgh);
        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          conv1d_tiled_caching(item,
                               tile.template get_multi_ptr<sycl::access::decorated::no>().get(),
                               mask, d_a, d_b, input_width, mask_width);
        });
      });
    }
    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time of conv1d-tiled-caching kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    q.memcpy(b, d_b, size_bytes).wait();
    reference(a, b, h_mask, input_width, mask_width);
  }

  free(a);
  free(b);
  sycl::free(mask, q);
  sycl::free(d_a, q);
  sycl::free(d_b, q);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <input_width> <repeat>\n", argv[0]);
    return 1;
  }

  int input_width = atoi(argv[1]);
  // a multiple of MAX BLOCK_SIZE
  input_width = (input_width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE * MAX_BLOCK_SIZE;

  const int repeat = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  for (int mask_width = 3; mask_width < MAX_MASK_WIDTH; mask_width += 2) {
    printf("\n---------------------\n");
    printf("Mask width: %d\n", mask_width); 

    printf("1D convolution (FP64)\n");
    conv1D<double>(q, input_width, mask_width, repeat);

    printf("1D convolution (FP32)\n");
    conv1D<float>(q, input_width, mask_width, repeat);

    printf("1D convolution (INT16)\n");
    conv1D<int16_t>(q, input_width, mask_width, repeat);
  }

  return 0;
}
