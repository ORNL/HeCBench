#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <hip/hip_runtime.h>
#include "topk_per_row_kernels.h"
#include <cstdlib>

__device__ __forceinline__ uint32_t xorshift32(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

__device__ __forceinline__ float rand_uniform(uint32_t& state) {
  return (xorshift32(state) & 0x00FFFFFF) / 16777216.0f;
}

__device__ float randn(uint32_t& state) {
  float u1 = rand_uniform(state);
  float u2 = rand_uniform(state);

  float r = sqrtf(-2.0f * logf(u1 + 1e-7f));
  float theta = 2.0f * 3.14159265358979323846f * u2;

  return r * cosf(theta);
}

// x = arange(hidden_size).repeat(batch_size, 1)

__global__ void init_x(float* x, int32_t batch_size, int32_t hidden_size)
{
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t total = batch_size * hidden_size;

  if (idx >= total) return;

  int32_t col = idx % hidden_size;
  x[idx] = (float)col;
}

// torch.randperm per row

__global__ void shuffle_rows(float* x, int32_t batch_size, int32_t hidden_size,
                             uint32_t seed)
{
  int32_t b = blockIdx.x;
  if (b >= batch_size) return;

  uint32_t state = seed ^ b;
  float* row = x + b * hidden_size;

  for (int32_t i = hidden_size - 1; i > 0; i--) {
    int32_t j = xorshift32(state) % (i + 1);

    float tmp = row[i];
    row[i] = row[j];
    row[j] = tmp;
  }
}

int32_t main(int32_t argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <batch_size> <repeat>\n", argv[0]);
    return 1;
  }
  const int32_t batch_size = atoi(argv[1]);
  const int32_t repeat = atoi(argv[2]);

  const int32_t HIDDENSIZES[] = {3072, 4096, 8192, 16384, 32768, 65536, 131072};
  const int32_t TOPKS[] = {2048, 1024};

  for (int32_t hidden_size : HIDDENSIZES) {
    for (int32_t topk : TOPKS) {
      printf("\nbatch size: %d, hidden size: %d, topk: %d\n", batch_size, hidden_size, topk);
      int32_t total = batch_size * hidden_size;

      float* d_x;
      int32_t*   d_topk_ids;
      float* d_topk_value;

      GPU_CHECK(hipMalloc(&d_x, total * sizeof(float)));
      GPU_CHECK(hipMalloc(&d_topk_ids, batch_size * topk * sizeof(int32_t)));
      GPU_CHECK(hipMalloc(&d_topk_value, batch_size * topk * sizeof(float)));
      std::vector<float> h_x(total);
      std::vector<int32_t> h_ids(batch_size * topk);
      std::vector<float> h_out(batch_size * topk);

      const int64_t stride0 = -1, // stride 0 will be hidden_size
                    stride1 = 1;

      int32_t threads = 256;
      int32_t blocks  = (total + threads - 1) / threads;

      init_x<<<blocks, threads>>>(d_x, batch_size, hidden_size);
      shuffle_rows<<<batch_size, 1>>>(d_x, batch_size, hidden_size, 5678);

      GPU_CHECK(hipMemcpy(h_x.data(), d_x, total * sizeof(float), hipMemcpyDeviceToHost));

      // warmup
      for (int32_t i = 0; i < 100; i++) {
        topk_radix<float, int32_t>(d_x, d_topk_ids, d_topk_value, topk,
                                   true, nullptr, nullptr, stride0, stride1,
                                   batch_size, hidden_size);
      }

      auto start = std::chrono::steady_clock::now();

      for (int32_t i = 0; i < repeat; i++) {
        topk_radix<float, int32_t>(d_x, d_topk_ids, d_topk_value, topk,
                                   true, nullptr, nullptr, stride0, stride1,
                                   batch_size, hidden_size);
      }

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of topk : %f (us)\n", (time * 1e-3f) / repeat);

      GPU_CHECK(hipMemcpy(h_out.data(), d_topk_value, batch_size * topk * sizeof(float), hipMemcpyDeviceToHost));
      GPU_CHECK(hipMemcpy(h_ids.data(), d_topk_ids, batch_size * topk * sizeof(int32_t), hipMemcpyDeviceToHost));

      bool ok = true;
      
      for (int32_t b = 0; b < batch_size; b++) {
        // partially sorted inputs from the largest
        auto start_it = h_x.begin() + b * hidden_size;
        auto end_it = h_x.begin() + (b+1) * hidden_size;
        std::partial_sort(start_it, start_it + topk, end_it, std::greater<float>());

        // sort device result
        auto o_start_it = h_out.begin() + b * topk;
        auto o_end_it = h_out.begin() + (b+1) * topk;
        std::sort(o_start_it, o_end_it, std::greater<float>());
        ok &= std::equal(start_it, start_it + topk, o_start_it);
        if (!ok) break;
      }

      printf("%s\n", ok ? "PASS" : "FAIL");
      if (!ok) exit(1);

      GPU_CHECK(hipFree(d_x));
      GPU_CHECK(hipFree(d_topk_ids));
      GPU_CHECK(hipFree(d_topk_value));
    }
  }

  return 0;
}
