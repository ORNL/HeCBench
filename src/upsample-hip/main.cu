#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "reference.h"
#include "common.h"

inline void gpu_check(hipError_t error, const char *file, int line) {
  if (error != hipSuccess) {
    printf("[HIP ERROR] at file %s:%d:\n%s\n", file, line,
           hipGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}
#define GPU_CHECK(err) (gpu_check(err, __FILE__, __LINE__))

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeat, Kernel kernel, KernelArgs&&... kernel_args) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    kernel(std::forward<KernelArgs>(kernel_args)...);
  }
  GPU_CHECK(hipDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-6f) / repeat;
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference,
                     const char* name, std::uint64_t num_elements,
                     T tolerance=1e-4, int n_print=5, int check_all=0)
{
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    GPU_CHECK(hipMemcpy(out_gpu, device_result, num_elements * sizeof(D), hipMemcpyDeviceToHost));
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

__global__ void upsample_forward_kernel(
    const float* __restrict__ x,
          float* __restrict__ out,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W)
{
  auto flat_idx = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
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

__global__ void upsample_forward_kernel2(
    const float* __restrict__ x,
          float* __restrict__ out,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W)
{
  auto in_x = blockIdx.x * blockDim.x + threadIdx.x;
  auto in_y = blockIdx.y * blockDim.y + threadIdx.y;
  auto bc   = blockIdx.z;

  if (in_x >= W || in_y >= H) return;

  auto b = bc / C;
  auto c = bc % C;

  auto H_out = H * 2;
  auto W_out = W * 2;

  float val = x[(b * C  + c) * H * W + in_y * W + in_x];

  float* out_base = out + (b * C + c) * H_out * W_out + (in_y * 2) * W_out + (in_x * 2);

  reinterpret_cast<float2*>(out_base)[0] = make_float2(val, val);
  reinterpret_cast<float2*>(out_base + W_out)[0] = make_float2(val, val);
}

__global__ void upsample_backward_kernel(
    const float* __restrict__ dout,
          float* __restrict__ dx,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W)
{
  auto flat_idx = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
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

__global__ void upsample_backward_kernel2(
    const float* __restrict__ dout,
          float* __restrict__ dx,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W)
{
  auto in_x = blockIdx.x * blockDim.x + threadIdx.x;
  auto in_y = blockIdx.y * blockDim.y + threadIdx.y;
  auto bc   = blockIdx.z;

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

  auto row0 = *(reinterpret_cast<const float2*>(dout_base));
  auto row1 = *(reinterpret_cast<const float2*>(dout_base + W_out));

  auto grad = row0.x + row0.y + row1.x + row1.y;

  dx[(b * C  + c) * H * W + in_y * W + in_x] = grad;
}

void upsample_forward1(
    float* out,
    const float* x,
    int B, int C, int H, int W,
    int block_size)
{
  int gridDim = ceil_div(B * C * H * W, block_size);
  upsample_forward_kernel<<<gridDim, block_size>>>(x, out, B, C, H, W);
}

void upsample_forward2(
    float* out,
    const float* x,
    int B, int C, int H, int W,
    int block_size_x,
    int block_size_y)
{
  dim3 block(block_size_x, block_size_y, 1);
  dim3 grid(ceil_div(W, block_size_x), ceil_div(H, block_size_y), B * C);
  upsample_forward_kernel2<<<grid, block>>>(x, out, B, C, H, W);
}

void upsample_backward1(
    float* dx,
    const float* dout,
    int B, int C, int H, int W,
    int block_size)
{
  int gridDim = ceil_div(B * C * H * W, block_size);
  upsample_backward_kernel<<<gridDim, block_size>>>(dout, dx, B, C, H, W);
}

void upsample_backward2(
    float* dx,
    const float* dout,
    int B, int C, int H, int W,
    int block_size_x,
    int block_size_y)
{
  dim3 block(block_size_x, block_size_y, 1);
  dim3 grid(ceil_div(W, block_size_x), ceil_div(H, block_size_y), B * C);
  upsample_backward_kernel2<<<grid, block>>>(dout, dx, B, C, H, W);
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

  float *d_x, *d_out, *d_dout, *d_dx;
  GPU_CHECK(hipMalloc(&d_x, S * sizeof(float)));
  GPU_CHECK(hipMalloc(&d_out, S * 4 * sizeof(float)));
  GPU_CHECK(hipMalloc(&d_dout, S * 4 * sizeof(float)));
  GPU_CHECK(hipMalloc(&d_dx, S * sizeof(float)));
  GPU_CHECK(hipMemcpy(d_x, x, S * sizeof(float), hipMemcpyHostToDevice));
  GPU_CHECK(hipMemcpy(d_dout, dout, S * 4 * sizeof(float), hipMemcpyHostToDevice));

  int block_sizes[] = {32, 64, 128, 256, 512, 1024};
  printf("Checking forward pass\n");
  for (int block_size : block_sizes) {
    printf("Checking block size %d\n", block_size);
    upsample_forward1(d_out, d_x, B, C, H, W, block_size);
    validate_result(d_out, out, "out", B * C * (H/2) * (W/2));
  }
  printf("Forward1 pass: all results match\n\n");

  printf("Forward1 pass benchmarks:\n");
  for (int block_size : block_sizes) {

    auto elapsed_time = benchmark_kernel(repeat, upsample_forward1, d_out, d_x, B, C, H, W, block_size);

    float tflops = (float)S / elapsed_time * 1e3f / 1e12f;
    printf("block_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
  }

  printf("\n─────────────────────────────────────────────────────\n");
  int block2D_sizes[] = {8, 16, 32};
  for (int block_size : block2D_sizes) {
    printf("Checking block size %d\n", block_size);
    upsample_forward2(d_out, d_x, B, C, H, W, block_size, block_size);
    validate_result(d_out, out, "out", B * C * (H/2) * (W/2));
  }
  printf("Forward2 pass: all results match\n\n");
  printf("Forward2 pass benchmarks:\n");
  for (int block_size : block2D_sizes) {

    auto elapsed_time = benchmark_kernel(repeat, upsample_forward2, d_out, d_x, B, C, H, W, block_size, block_size);

    float tflops = (float)S / elapsed_time * 1e3f / 1e12f;
    printf("block2D_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
  }

  printf("\n─────────────────────────────────────────────────────\n");
  printf("Checking backward pass\n");
  upsample_backward_reference(dout, dx, B, C, H, W);
  for (int block_size : block_sizes) {
    printf("Checking block size %d\n", block_size);
    upsample_backward1(d_dx, d_dout, B, C, H, W, block_size);
    validate_result(d_dx, dx, "dx", S);
  }
  printf("Backward pass: all results match\n\n");
  printf("All results match. Starting benchmarks.\n\n");

  printf("\nBackward pass benchmarks:\n");
  for (int block_size : block_sizes) {

    auto elapsed_time = benchmark_kernel(repeat, upsample_backward1, d_dx, d_dout, B, C, H, W, block_size);

    float tflops = (float)S / elapsed_time * 1e3f / 1e12f;
    printf("block_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
  }

  printf("\n─────────────────────────────────────────────────────\n");
  printf("Checking backward2 pass\n");
  for (int block_size : block2D_sizes) {
    printf("Checking block size %d\n", block_size);
    upsample_backward2(d_dx, d_dout, B, C, H, W, block_size, block_size);
    validate_result(d_dx, dx, "dx", S);
  }
  printf("Backward2 pass: all results match\n\n");
  printf("All results match. Starting benchmarks.\n\n");

  printf("\nBackward2 pass benchmarks:\n");
  for (int block_size : block2D_sizes) {

    auto elapsed_time = benchmark_kernel(repeat, upsample_backward2, d_dx, d_dout, B, C, H, W, block_size, block_size);

    float tflops = (float)S / elapsed_time * 1e3f / 1e12f;
    printf("block2D_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
  }

  GPU_CHECK(hipFree(d_x));
  GPU_CHECK(hipFree(d_out));
  GPU_CHECK(hipFree(d_dout));
  GPU_CHECK(hipFree(d_dx));

  free(x);
  free(out);
  free(dout);
  free(dx);

  return 0;
}
