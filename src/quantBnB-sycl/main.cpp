#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <sycl/sycl.hpp>
#include "block_load.h"
#include "block_store.h"
#include "code.h"

template <int STOCHASTIC>
uint8_t
dQuantize(float* smem_code, const float rand, float x)
{
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = -1.0f;
    float upper = 1.0f;

    float val = smem_code[pivot];
    // i>>=1 = {32, 16, 8, 4, 2, 1}
    for(int i = 64; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
        }
        val = smem_code[pivot];
    }

    if(upper_pivot == 255)
        upper = smem_code[upper_pivot];
    if(lower_pivot == 0)
        lower = smem_code[lower_pivot];

    if(!STOCHASTIC)
    {
      if(x > val)
      {
        float midpoint = (upper+val)*0.5f;
        if(x > midpoint)
        {
          return upper_pivot;
        }
        else
          return pivot;
      }
      else
      {
        float midpoint = (lower+val)*0.5f;
        if(x < midpoint)
          return lower_pivot;
        else
          return pivot;
      }
    }
    else
    {
      if(x > val)
      {
        float dist_to_upper = sycl::fabs(upper - x);
        float dist_full = upper-val;
        if(rand >= dist_to_upper/dist_full) return upper_pivot;
        else return pivot;
      }
      else
      {
        float dist_to_lower = sycl::fabs(lower - x);
        float dist_full = val-lower;
        if(rand >= dist_to_lower/dist_full) return lower_pivot;
        else return pivot;
      }
    }
}


#define NUM 4

template<int TH, int NUM_BLOCK>
void kQuantize(const float *__restrict__ code,
               const float * __restrict__ A,
               uint8_t *out, const int n,
               const sycl::nd_item<3> &item)
{
  const size_t bid = item.get_group(2);
  const int tid = item.get_local_id(2);
  const int n_full = item.get_group_range(2) * NUM_BLOCK;
  const int base_idx = bid * NUM_BLOCK;

  float vals[NUM];
  uint8_t qvals[NUM];

  typedef BlockLoad<float, TH, NUM> LoadFloat;
  typedef BlockStore<uint8_t, TH, NUM> StoreChar;

  sycl::multi_ptr<float[256], sycl::access::address_space::local_space> p1 =
      sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(item.get_group());
  float *smem_code = *p1;

  sycl::multi_ptr<typename LoadFloat::TempStorage, sycl::access::address_space::local_space> p2 =
      sycl::ext::oneapi::group_local_memory_for_overwrite<typename LoadFloat::TempStorage>(item.get_group());
  typename LoadFloat::TempStorage loadf_storage = *p2;

  sycl::multi_ptr<typename StoreChar::TempStorage, sycl::access::address_space::local_space> p3 =
      sycl::ext::oneapi::group_local_memory_for_overwrite<typename StoreChar::TempStorage>(item.get_group());
  typename StoreChar::TempStorage storec_storage = *p3;

  for (int i = tid; i < 256; i += item.get_local_range(2))
  {
    smem_code[i] = code[i];
  }

  for (int i = base_idx; i < n; i += n_full)
  {
      int valid_items = sycl::min(n - i, NUM_BLOCK);

      LoadFloat(loadf_storage, item).Load(&(A[i]), vals, valid_items);

      item.barrier(sycl::access::fence_space::local_space);

      #pragma unroll
      for(int j = 0; j < NUM; j++)
          qvals[j] = dQuantize<0>(smem_code, 0.0f, vals[j]);

      StoreChar(storec_storage, item).Store(&(out[i]), qvals, valid_items);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t n = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  const size_t A_size = n * sizeof(float);
  const size_t out_size = n * sizeof(uint8_t);
  const size_t code_size = 256 * sizeof(float); // code.h

  std::vector<float> A(n);
  std::vector<uint8_t> out(n), ref(n);

  std::mt19937 gen{19937};
 
  std::normal_distribution<float> d{0.f, 1.f};

  for (size_t i = 0; i < n; i++) {
    A[i] = d(gen); 
    ref[i] = dQuantize<0>(code, 0.0f, A[i]);
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_A, *d_code;
  d_A = (float *)sycl::malloc_device(A_size, q);
  q.memcpy(d_A, A.data(), A_size).wait();

  d_code = (float *)sycl::malloc_device(code_size, q);
  q.memcpy(d_code, code, code_size).wait();

  uint8_t *d_out;
  d_out = (uint8_t *)sycl::malloc_device(out_size, q);

  const int block_size = 256;

  sycl::range<3> grid(1, 1, (n + block_size - 1) / block_size);
  sycl::range<3> block(1, 1, block_size);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
        [=](sycl::nd_item<3> item) {
        kQuantize<block_size, block_size>(d_code, d_A, d_out, n, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kQuantize kernel with block size %d: %f (us)\n",
          block_size, (time * 1e-3f) / repeat);

  q.memcpy(out.data(), d_out, out_size).wait();

  printf("%s\n", out == ref ? "PASS" : "FAIL");

  sycl::free(d_A, q);
  sycl::free(d_code, q);
  sycl::free(d_out, q);
  if (out != ref) return 1;
  return 0;
}
