#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include "reference.h"
#include "common.h"

template<class Kernel, class... KernelArgs>
float benchmark_kernel(sycl::queue &q, int repeat, Kernel kernel, KernelArgs&&... kernel_args) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    kernel(std::forward<KernelArgs>(kernel_args)...);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-6f) / repeat;
}

template<class D, class T>
void validate_result(sycl::queue &q,
                     D* device_result, const T* cpu_reference,
                     const char* name, std::uint64_t num_elements,
                     T tolerance=1e-4, int n_print=5, int check_all=0)
{
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    q.memcpy(out_gpu, device_result, num_elements * sizeof(D)).wait();
    int nfaults = 0;
    for (uint64_t i = 0; i < num_elements; i++) {
      if (std::fabs(cpu_reference[i] - (T)out_gpu[i]) > tolerance && std::isfinite(cpu_reference[i])) {
        printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
        nfaults++;
        if (nfaults >= max_int(10, n_print)) {
          free(out_gpu);
        }
      }
    }

    free(out_gpu);
}

void upsample_forward_kernel(
    const float* __restrict__ x,
          float* __restrict__ out,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W,
    sycl::nd_item<3> &item)
{
  auto flat_idx = item.get_global_id(2);
  auto img_size = H * W;
  if (flat_idx >= B * C * img_size) {
    return;
  }

  auto H_out = H*2;
  auto W_out = W*2;
  auto img_out_size = H_out * W_out;

  auto b = flat_idx / (C * img_size);
  auto c = (flat_idx / img_size) % C;
  auto i = (flat_idx / W) % H;
  auto j = flat_idx % W;

  x   += b * C * img_size + c * img_size + i * W + j;
  out += b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;

  float x_val = x[0];

  for (int ii = 0; ii < 2; ii++) {
    for (int jj = 0; jj < 2; jj++) {
      out[ii * W_out + jj] = x_val;
    }
  }
}

void upsample_forward_kernel2(
    const float* __restrict__ x,
          float* __restrict__ out,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W,
    sycl::nd_item<3> &item)
{
  auto in_x = item.get_global_id(2);
  auto in_y = item.get_global_id(1);
  auto bc = item.get_group(0);

  if (in_x >= W || in_y >= H) return;

  auto b = bc / C;
  auto c = bc % C;

  auto H_out = H * 2;
  auto W_out = W * 2;

  float val = x[(b * C  + c) * H * W + in_y * W + in_x];

  float* out_base = out + (b * C + c) * H_out * W_out + (in_y * 2) * W_out + (in_x * 2);

  reinterpret_cast<sycl::float2 *>(out_base)[0] = sycl::float2(val, val);
  reinterpret_cast<sycl::float2 *>(out_base + W_out)[0] = sycl::float2(val, val);
}

void upsample_backward_kernel(
    const float* __restrict__ dout,
          float* __restrict__ dx,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W,
    sycl::nd_item<3> &item)
{
  auto flat_idx = item.get_global_id(2);
  auto img_in_size = H * W;
  if (flat_idx >= B * C * img_in_size) {
    return;
  }

  auto H_out = H*2;
  auto W_out = W*2;
  auto img_out_size = H_out * W_out;

  auto b = flat_idx / (C * img_in_size);
  auto c = (flat_idx / img_in_size) % C;
  auto i = (flat_idx / W) % H;
  auto j = flat_idx % W;

  // move pointers
  dx += b * C * img_in_size + c * img_in_size + i * W + j;
  dout += b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;

  float dout_sum = 0.0f;
  for (int ii = 0; ii < 2; ii++) {
    for (int jj = 0; jj < 2; jj++) {
      dout_sum += dout[ii * W_out + jj];
    }
  }
  dx[0] = dout_sum;
}

void upsample_backward_kernel2(
    const float* __restrict__ dout,
          float* __restrict__ dx,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W,
    sycl::nd_item<3> &item)
{
  auto in_x = item.get_global_id(2);
  auto in_y = item.get_global_id(1);
  auto bc = item.get_group(0);

  if (in_x >= W || in_y >= H) return;

  auto b = bc / C;
  auto c = bc % C;

  auto W_out = W * 2;
  auto H_out = H * 2;

  // Base pointer into dout for this thread's 2x2 tile
  const auto dout_base = dout + b * (C * H_out * W_out)
                              + c * (H_out * W_out)
                              + (in_y * 2) * W_out
                              + (in_x * 2);

  auto row0 = *(reinterpret_cast<const sycl::float2 *>(dout_base));
  auto row1 = *(reinterpret_cast<const sycl::float2 *>(dout_base + W_out));

  auto grad = row0.x() + row0.y() + row1.x() + row1.y();

  dx[(b * C  + c) * H * W + in_y * W + in_x] = grad;
}

void upsample_forward1(
    sycl::queue &q,
    float* out,
    const float* x,
    int B, int C, int H, int W,
    int block_size)
{
  int gridDim = ceil_div(B * C * H * W, block_size);
  q.parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, gridDim * block_size),
                      sycl::range<3>(1, 1, block_size)),
    [=](sycl::nd_item<3> item) {
      upsample_forward_kernel(x, out, B, C, H, W, item);
  });
}

void upsample_forward2(
    sycl::queue &q,
    float* out,
    const float* x,
    int B, int C, int H, int W,
    int block_size_x,
    int block_size_y)
{
  sycl::range<3> lws (1, block_size_y, block_size_x);
  sycl::range<3> gws (B * C, ceil_div(H, block_size_y) * block_size_y,
                             ceil_div(W, block_size_x) * block_size_x);
  q.parallel_for(
    sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
      upsample_forward_kernel2(x, out, B, C, H, W, item);
  });
}

void upsample_backward1(
    sycl::queue &q,
    float* dx,
    const float* dout,
    int B, int C, int H, int W,
    int block_size)
{
  int gridDim = ceil_div(B * C * H * W, block_size);
  q.parallel_for(
    sycl::nd_range<3>(sycl::range<3>(1, 1, gridDim * block_size),
                      sycl::range<3>(1, 1, block_size)),
    [=](sycl::nd_item<3> item) {
    upsample_backward_kernel(dout, dx, B, C, H, W, item);
  });
}

void upsample_backward2(
    sycl::queue &q,
    float* dx,
    const float* dout,
    int B, int C, int H, int W,
    int block_size_x,
    int block_size_y)
{
  sycl::range<3> lws (1, block_size_y, block_size_x);
  sycl::range<3> gws (B * C, ceil_div(H, block_size_y) * block_size_y,
                             ceil_div(W, block_size_x) * block_size_x);
  q.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
    upsample_backward_kernel2(dout, dx, B, C, H, W, item);
  });
}

int main(int argc, char **argv) {
  if (argc != 6) {
    printf("Usage: %s <batch size> <number of channels> <height> <width> <repeat>\n", argv[0]);
    return 1;
  }
  int B = atoi(argv[1]);
  int C = atoi(argv[2]);
  int H = atoi(argv[3]);
  int W = atoi(argv[4]);
  int repeat = atoi(argv[5]);

  srand(0);
  uint64_t S = (uint64_t)B * C * H * W;
  float* x = make_random_float(S);
  float* dout = make_random_float(S * 4);

  float* out = (float*)malloc(S * 4 * sizeof(float));
  float* dx = (float*)malloc(S * sizeof(float));

  upsample_forward_reference(x, out, B, C, H, W);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_x, *d_out, *d_dout, *d_dx;
  d_x = sycl::malloc_device<float>(S, q);
  d_out = sycl::malloc_device<float>(S * 4, q);
  d_dout = sycl::malloc_device<float>(S * 4, q);
  d_dx = sycl::malloc_device<float>(S, q);
  q.memcpy(d_x, x, S * sizeof(float));
  q.memcpy(d_dout, dout, S * 4 * sizeof(float));

  int block_sizes[] = {32, 64, 128, 256, 512, 1024};
  printf("Checking forward pass\n");
  for (int block_size : block_sizes) {
    printf("Checking block size %d\n", block_size);
    upsample_forward1(q, d_out, d_x, B, C, H, W, block_size);
    validate_result(q, d_out, out, "out", B * C * (H/2) * (W/2));
  }
  printf("Forward1 pass: all results match\n\n");

  printf("Forward1 pass benchmarks:\n");
  for (int block_size : block_sizes) {

    auto elapsed_time = benchmark_kernel(q, repeat, upsample_forward1, q, d_out, d_x, B, C, H, W, block_size);

    float tflops = (float)S / elapsed_time * 1e3f / 1e12f;
    printf("block_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
  }

  printf("\n─────────────────────────────────────────────────────\n");
  int block2D_sizes[] = {8, 16, 32};
  for (int block_size : block2D_sizes) {
    printf("Checking block size %d\n", block_size);
    upsample_forward2(q, d_out, d_x, B, C, H, W, block_size, block_size);
    validate_result(q, d_out, out, "out", B * C * (H/2) * (W/2));
  }
  printf("Forward2 pass: all results match\n\n");
  printf("Forward2 pass benchmarks:\n");
  for (int block_size : block2D_sizes) {

    auto elapsed_time = benchmark_kernel(q, repeat, upsample_forward2, q, d_out, d_x, B, C, H, W, block_size, block_size);

    float tflops = (float)S / elapsed_time * 1e3f / 1e12f;
    printf("block2D_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
  }

  printf("\n─────────────────────────────────────────────────────\n");
  printf("Checking backward pass\n");
  upsample_backward_reference(dout, dx, B, C, H, W);
  for (int block_size : block_sizes) {
    printf("Checking block size %d\n", block_size);
    upsample_backward1(q, d_dx, d_dout, B, C, H, W, block_size);
    validate_result(q, d_dx, dx, "dx", S);
  }
  printf("Backward pass: all results match\n\n");
  printf("All results match. Starting benchmarks.\n\n");

  printf("\nBackward pass benchmarks:\n");
  for (int block_size : block_sizes) {

    auto elapsed_time = benchmark_kernel(q, repeat, upsample_backward1, q, d_dx, d_dout, B, C, H, W, block_size);

    float tflops = (float)S / elapsed_time * 1e3f / 1e12f;
    printf("block_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
  }

  printf("\n─────────────────────────────────────────────────────\n");
  printf("Checking backward2 pass\n");
  for (int block_size : block2D_sizes) {
    printf("Checking block size %d\n", block_size);
    upsample_backward2(q, d_dx, d_dout, B, C, H, W, block_size, block_size);
    validate_result(q, d_dx, dx, "dx", S);
  }
  printf("Backward2 pass: all results match\n\n");
  printf("All results match. Starting benchmarks.\n\n");

  printf("\nBackward2 pass benchmarks:\n");
  for (int block_size : block2D_sizes) {

    auto elapsed_time = benchmark_kernel(q, repeat, upsample_backward2, q, d_dx, d_dout, B, C, H, W, block_size, block_size);

    float tflops = (float)S / elapsed_time * 1e3f / 1e12f;
    printf("block2D_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
  }

  sycl::free(d_x, q);
  sycl::free(d_out, q);
  sycl::free(d_dout, q);
  sycl::free(d_dx, q);

  free(x);
  free(out);
  free(dout);
  free(dx);

  return 0;
}
