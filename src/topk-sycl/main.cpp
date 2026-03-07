#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include "topk_per_row_kernels.h"
#include <cstdlib>

inline uint32_t xorshift32(uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

inline float rand_uniform(uint32_t &state) {
  return (xorshift32(state) & 0x00FFFFFF) / 16777216.0f;
}

float randn(uint32_t& state) {
  float u1 = rand_uniform(state);
  float u2 = rand_uniform(state);

  float r = sycl::sqrt(-2.0f * sycl::log(u1 + 1e-7f));
  float theta = 2.0f * 3.14159265358979323846f * u2;

  return r * sycl::cos(theta);
}

// x = arange(hidden_size).repeat(batch_size, 1)

void init_x(float* x, int32_t batch_size, int32_t hidden_size, sycl::nd_item<3> &item)
{
  int32_t idx = item.get_global_id(2);
  int32_t total = batch_size * hidden_size;

  if (idx >= total) return;

  int32_t col = idx % hidden_size;
  x[idx] = (float)col;
}

// torch.randperm per row

void shuffle_rows(float* x, int32_t batch_size, int32_t hidden_size,
                  uint32_t seed, sycl::nd_item<3> &item)
{
  int32_t b = item.get_group(2);
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

int32_t main(int32_t argc, char *argv[]) try {
  if (argc != 3) {
    printf("Usage: %s <batch_size> <repeat>\n", argv[0]);
    return 1;
  }
  const int32_t batch_size = atoi(argv[1]);
  const int32_t repeat = atoi(argv[2]);

  const int32_t HIDDENSIZES[] = {3072, 4096, 8192, 16384, 32768, 65536, 131072};
  const int32_t TOPKS[] = {2048, 1024};

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  for (int32_t hidden_size : HIDDENSIZES) {
    for (int32_t topk : TOPKS) {
      printf("\nbatch size: %d, hidden size: %d, topk: %d\n", batch_size, hidden_size, topk);
      int32_t total = batch_size * hidden_size;

      float *d_x = sycl::malloc_device<float>(total, q);
      int32_t *d_topk_ids = sycl::malloc_device<int32_t>(batch_size * topk, q);
      float *d_topk_value = sycl::malloc_device<float>(batch_size * topk, q);
      std::vector<float> h_x(total);
      std::vector<int32_t> h_ids(batch_size * topk);
      std::vector<float> h_out(batch_size * topk);

      const int64_t stride0 = -1, // stride 0 will be hidden_size
                    stride1 = 1;

      int32_t threads = 256;
      int32_t blocks  = (total + threads - 1) / threads;

      q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks * threads),
                          sycl::range<3>(1, 1, threads)),
        [=](sycl::nd_item<3> item) {
            init_x(d_x, batch_size, hidden_size, item);
          });
      q.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size),
                            sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item) {
              shuffle_rows(d_x, batch_size, hidden_size, 5678, item);
          });

      q.memcpy(h_x.data(), d_x, total * sizeof(float)).wait();

      // warmup
      for (int32_t i = 0; i < 100; i++) {
        topk_radix<float, int32_t>(d_x, d_topk_ids, d_topk_value, topk,
                                   true, nullptr, nullptr, stride0, stride1,
                                   batch_size, hidden_size, q);
      }

      auto start = std::chrono::steady_clock::now();

      for (int32_t i = 0; i < repeat; i++) {
        topk_radix<float, int32_t>(d_x, d_topk_ids, d_topk_value, topk,
                                   true, nullptr, nullptr, stride0, stride1,
                                   batch_size, hidden_size, q);
      }

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of topk : %f (us)\n", (time * 1e-3f) / repeat);

      q.memcpy(h_out.data(), d_topk_value, batch_size * topk * sizeof(float));
      q.memcpy(h_ids.data(), d_topk_ids, batch_size * topk * sizeof(int32_t));
      q.wait();

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

      sycl::free(d_x, q);
      sycl::free(d_topk_ids, q);
      sycl::free(d_topk_value, q);
    }
  }

  if (!ok) return 1;
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  //std::exit(1);
}
