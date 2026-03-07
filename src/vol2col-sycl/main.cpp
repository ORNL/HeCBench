#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

#define threadsPerBlock 512

// Kernel for fast unfold+copy on volumes
template <typename T>
void vol2col_kernel(
    sycl::nd_item<1> &item,
    const int64_t range,
    const T* data_vol,
    const int depth,
    const int height,
    const int width,
    const int ksize_t,
    const int ksize_h,
    const int ksize_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int depth_col,
    const int height_col,
    const int width_col,
    T* data_col)
{
  for (int64_t n = item.get_global_id(0); n < range;
               n += item.get_group_range(0) * item.get_local_range(0)) {
    int w_out = n % width_col;
    int64_t index = n / width_col;
    int h_out = index % height_col;
    index /= height_col;
    int t_out = index % depth_col;
    int channel_in = index / depth_col;
    int channel_out = channel_in * ksize_t * ksize_h * ksize_w;
    int t_in = t_out * stride_t - pad_t;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    auto v = data_vol + ((channel_in * depth + t_in) * height + h_in) * width + w_in;
    auto c = data_col + ((channel_out * depth_col + t_out) * height_col + h_out) * width_col + w_out;
    for (int i = 0; i < ksize_t; ++i) {
      for (int j = 0; j < ksize_h; ++j) {
        for (int k = 0; k < ksize_w; ++k) {
          int t = t_in + i * dilation_t;
          int h = h_in + j * dilation_h;
          int w = w_in + k * dilation_w;
          *c = (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height && w < width)
              ? v[i * dilation_t * height * width +
                  j * dilation_h * width + k * dilation_w]
              : static_cast<T>(0);
          c += depth_col * height_col * width_col;
        }
      }
    }
  }
}

template <typename T, typename accT>
void col2vol_kernel(
    sycl::nd_item<1> &item,
    const int64_t n,
    const T* data_col,
    const int depth,
    const int height,
    const int width,
    const int kernel_t,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int depth_col,
    const int height_col,
    const int width_col,
    T* data_vol)
{
  for (int64_t index = item.get_global_id(0); index < n;
               index += item.get_group_range(0) * item.get_local_range(0)) {
    accT val = static_cast<accT>(0);
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int t_im = (index / width / height) % depth + pad_t;
    const int c_im = index / (width * height * depth);
    auto kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    auto kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    auto kernel_extent_t = (kernel_t - 1) * dilation_t + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = sycl::min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = sycl::min(h_im / stride_h + 1, height_col);
    const int t_col_start =
        (t_im < kernel_extent_t) ? 0 : (t_im - kernel_extent_t) / stride_t + 1;
    const int t_col_end = sycl::min(t_im / stride_t + 1, depth_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int t_col = t_col_start; t_col < t_col_end; t_col += 1) {
      for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
        for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
          int64_t t_k = (t_im - t_col * stride_t);
          int64_t h_k = (h_im - h_col * stride_h);
          int64_t w_k = (w_im - w_col * stride_w);
          if (t_k % dilation_t == 0 && h_k % dilation_h == 0 &&
              w_k % dilation_w == 0) {
            t_k /= dilation_t;
            h_k /= dilation_h;
            w_k /= dilation_w;
            const int64_t idx_k =
                ((c_im * kernel_t + t_k) * kernel_h + h_k) * kernel_w + w_k;
            const int64_t data_col_index =
                ((idx_k * depth_col + t_col) *
                    height_col + h_col) *
                  width_col + w_col;
            val += data_col[data_col_index];
          }
        }
      }
    }
    data_vol[index] = static_cast<T>(val);
  }
}

int get_blocks (sycl::queue &q, int64_t n) {
  int64_t numBlocks = (n - threadsPerBlock) / threadsPerBlock + 1;

#ifdef SYCL_EXT_ONEAPI_MAX_WORK_GROUP_QUERY
  sycl::id<3> groups = q.get_device().get_info<
    sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>();
  int64_t blocksPerGrid = std::min((int64_t)groups[2], numBlocks);
#else
  int64_t blocksPerGrid = std::min((int64_t)0x7FFFFFFF, numBlocks);
#endif

  return blocksPerGrid;
}

template <typename T>
void eval (
    const int repeat,
    const int channels,
    const int depth,
    const int height,
    const int width,
    const int depth_col,
    const int height_col,
    const int width_col,
    const int ksize_t,
    const int ksize_h,
    const int ksize_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w)
{
  int64_t vol_size = (int64_t) channels * (2*pad_t+depth) * (2*pad_h+height) * (2*pad_w+width);
  int64_t col_size = ((int64_t) channels * ksize_t * ksize_h * ksize_w + 1) *
                    (depth_col+pad_t) * (height_col+pad_h) * (width_col+pad_w);

  int64_t vol_size_bytes = sizeof(T) * vol_size;
  int64_t col_size_bytes = sizeof(T) * col_size;

  T *h_data_vol = (T*) malloc (vol_size_bytes);
  T *h_data_col = (T*) malloc (col_size_bytes);
  T *h_data_vol_ref = (T*) malloc (vol_size_bytes);
  T *h_data_col_ref = (T*) malloc (col_size_bytes);

  srand(123);
  for (int64_t i = 0; i < vol_size; i++) {
    h_data_vol_ref[i] = h_data_vol[i] = rand() / (T) RAND_MAX;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  T *d_data_vol = sycl::malloc_device<T>(vol_size, q);
  q.memcpy(d_data_vol, h_data_vol, vol_size_bytes);

  T *d_data_col = sycl::malloc_device<T>(col_size, q);
  q.memset(d_data_col, 0, col_size_bytes);

  // each of "channels * depth_col * height_col * width_col"
  // blocks responsible for copying a single-channel grid.
  // We cast an operand to int64 so that the product will not overflow
  int64_t n = static_cast<int64_t>(channels) * depth_col * height_col * width_col;

  int blocksPerGrid = get_blocks(q, n);

  sycl::range<1> gws (blocksPerGrid * threadsPerBlock);
  sycl::range<1> lws (threadsPerBlock);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class v2c>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        vol2col_kernel<T>(
          item,
          n,
          d_data_vol,
          depth, height, width,
          ksize_t, ksize_h, ksize_w,
          pad_t, pad_h, pad_w,
          stride_t, stride_h, stride_w,
          dilation_t, dilation_h, dilation_w,
          depth_col, height_col, width_col,
          d_data_col);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of vol2col kernel: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(h_data_col, d_data_col, col_size_bytes).wait();

  // verify
  vol2col_reference<T>(
      h_data_vol,
      channels, depth, height, width,
      ksize_t, ksize_h, ksize_w,
      pad_t, pad_h, pad_w,
      stride_t, stride_h, stride_w,
      dilation_t, dilation_h, dilation_w,
      depth_col, height_col, width_col,
      h_data_col_ref);

  int error = memcmp(h_data_col_ref, h_data_col, col_size_bytes);
  printf("%s\n", error ? "FAIL" : "PASS");

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class c2v>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        col2vol_kernel<T, T>(
          item,
          n,
          d_data_col,
          depth, height, width,
          ksize_t, ksize_h, ksize_w,
          pad_t, pad_h, pad_w,
          stride_t, stride_h, stride_w,
          dilation_t, dilation_h, dilation_w,
          depth_col, height_col, width_col,
          d_data_vol);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of col2vol kernel: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(h_data_vol, d_data_vol, vol_size_bytes).wait();

  // verify
  col2vol_reference<T, T>(
      h_data_col_ref,
      channels, depth, height, width,
      ksize_t, ksize_h, ksize_w,
      pad_t, pad_h, pad_w,
      stride_t, stride_h, stride_w,
      dilation_t, dilation_h, dilation_w,
      depth_col, height_col, width_col,
      h_data_vol_ref);

  for (int64_t i = 0; i < vol_size; i++) {
    if (fabsf(h_data_vol_ref[i] - h_data_vol[i]) > 1e-3f) {
      error = 1;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  sycl::free(d_data_vol, q);
  sycl::free(d_data_col, q);
  free(h_data_vol);
  free(h_data_col);
  free(h_data_vol_ref);
  free(h_data_col_ref);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int channels = 4;
  int depth = 3;
  int height = 255;
  int width = 255;
  int pad_t = 1;
  int pad_h = 1;
  int pad_w = 1;
  int stride_t = 2;
  int stride_h = 2;
  int stride_w = 2;
  int dilation_t = 2;
  int dilation_h = 2;
  int dilation_w = 2;
  int depth_col = 3;
  int height_col = 255;
  int width_col = 255;

  for (int k = 1; k <= 9; k = k + 2) {
    printf("\nkernel size: %d\n", k);
    int ksize_t = k;
    int ksize_h = k;
    int ksize_w = k;

    eval<float> (repeat,
                 channels, depth, height, width,
                 depth_col, height_col, width_col,
                 ksize_t, ksize_h, ksize_w,
                 pad_t, pad_h, pad_w,
                 stride_t, stride_h, stride_w,
                 dilation_t, dilation_h, dilation_w);
  }

  if (error) return 1;
  if (error) return 1;
  return 0;
}
