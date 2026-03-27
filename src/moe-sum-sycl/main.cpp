#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include "kernels.h"

template <typename scalar_t>
int64_t moe_sum(sycl::queue &q,
                const scalar_t* input,   // [num_tokens, topk, hidden_size]
                     scalar_t* output,  // [num_tokens, hidden_size]
                const int hidden_size,
                const int num_tokens,
                const int topk,
                const int repeat)
{
  int block_size = std::min(hidden_size, 1024);
  sycl::range<1> gws(block_size * num_tokens);
  sycl::range<1> lws(block_size);

  auto kFn = [=](sycl::nd_item<1> item) {
    moe_sum_kernel<scalar_t, 2>(output, input, hidden_size, item);
  };

  // warmup
  for (int i = 0; i < 30; i++) {
    q.parallel_for(sycl::nd_range<1>(gws, lws), kFn);
  }
  q.wait();

  auto start = std::chrono::steady_clock::now();

  switch (topk) {
    case 2:
      for (int i = 0; i < repeat; i++)
        q.parallel_for(sycl::nd_range<1>(gws, lws), kFn);
      break;

    case 3:
      for (int i = 0; i < repeat; i++)
        q.parallel_for(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
            moe_sum_kernel<scalar_t, 3>(output, input, hidden_size, item);
        });
      break;

    case 4:
      for (int i = 0; i < repeat; i++)
        q.parallel_for(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
            moe_sum_kernel<scalar_t, 4>(output, input, hidden_size, item);
        });
      break;

    default:
      printf("Not implemented for topK value %d\n", topk);
      break;
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernel (TopK = %d): %f (us)\n",
         topk, (time * 1e-3f) / repeat);
  return time;
}

int64_t moe_sum_vec4(sycl::queue &q,
               const float* input,
                     float* output,
                const int hidden_size,
                const int num_tokens,
                const int topk,
                const int repeat)
{
  int block_size = std::min(hidden_size / 4, 1024);
  sycl::range<1> gws(block_size * num_tokens);
  sycl::range<1> lws(block_size);

  auto kFn = [=](sycl::nd_item<1> item) {
    moe_sum_kernel_vec4<2>(output, input, hidden_size, item);
  };

  // warmup
  for (int i = 0; i < 30; i++) {
    q.parallel_for(sycl::nd_range<1>(gws, lws), kFn);
  }
  q.wait();

  auto start = std::chrono::steady_clock::now();

  switch (topk) {
    case 2:
      for (int i = 0; i < repeat; i++)
        q.parallel_for(sycl::nd_range<1>(gws, lws), kFn);
      break;

    case 3:
      for (int i = 0; i < repeat; i++)
        q.parallel_for(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
            moe_sum_kernel_vec4<3>(output, input, hidden_size, item);
        });
      break;

    case 4:
      for (int i = 0; i < repeat; i++)
        q.parallel_for(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
            moe_sum_kernel_vec4<4>(output, input, hidden_size, item);
        });
      break;

    default:
      printf("Not implemented for topK value %d\n", topk);
      break;
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of vec4 kernel (TopK = %d): %f (us)\n",
         topk, (time * 1e-3f) / repeat);
  return time;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of tokens> <hidden size> <repeat>\n", argv[0]);
    return 1;
  }
  const int32_t num_tokens = atoi(argv[1]);
  const int32_t hidden_size = atoi(argv[2]);
  const int32_t repeat = atoi(argv[3]);

  if (hidden_size % 4) {
    printf("Hidden size is a multiple of four\n");
    return 1;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int64_t output_size_bytes = (int64_t)num_tokens * hidden_size * sizeof(float);
  float *d_output, *output, *output_vec4;
  d_output = (float *)sycl::malloc_device(output_size_bytes, q);
  output = (float*) malloc (output_size_bytes);
  output_vec4 = (float*) malloc (output_size_bytes);

  for (int topk = 2; topk <= 4; topk++) {
    const int64_t input_size_bytes = output_size_bytes * topk;

    float *d_input, *input;
    d_input = (float *)sycl::malloc_device(input_size_bytes, q);
    input = (float*) malloc (input_size_bytes);

    std::mt19937 gen(topk);
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for (int64_t i = 0; i < (int64_t)num_tokens * hidden_size * topk; i++) {
      input[i] = dis(gen);
    }
    q.memcpy(d_input, input, input_size_bytes).wait();

    int64_t nano_seconds = moe_sum(q, d_input, d_output, hidden_size, num_tokens, topk, repeat);

    q.memcpy(output, d_output, output_size_bytes).wait();

    float io_bytes = 1.f * repeat * (input_size_bytes + output_size_bytes);
    float bw = io_bytes / nano_seconds;

    printf("Kernel bandwidth: %f GB/s \n", bw);

    nano_seconds = moe_sum_vec4(q, d_input, d_output, hidden_size, num_tokens, topk, repeat);
    float bw_vec4 = io_bytes / nano_seconds;
    printf("Kernel(vec4) bandwidth: %f GB/s (%f%%)\n", bw_vec4, 100 * (bw_vec4 - bw) / bw);

    q.memcpy(output_vec4, d_output, output_size_bytes).wait();

    int32_t rc = memcmp(output, output_vec4, output_size_bytes);

    printf("%s\n", rc ? "FAIL" : "PASS");

    sycl::free(d_input, q);
    free(input);
  }
  sycl::free(d_output, q);
  free(output);
  free(output_vec4);
  return 0;
}
