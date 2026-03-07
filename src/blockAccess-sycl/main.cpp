#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include "utils.h"
#include "block_load.h"
#include "block_store.h"

namespace sycl_exp = sycl::ext::oneapi::experimental;

void reference (const float * __restrict__ A,
                unsigned char *out, const size_t n,
                const sycl::nd_item<3> &item)
{
  for (size_t idx = item.get_global_id(2);
       idx < n/4; idx += item.get_local_range(2) * item.get_group_range(2)) {
    const sycl::float4 v = reinterpret_cast<const sycl::float4*>(A)[idx];
    sycl::uchar4 o;
    o.x() = (int)v.x();
    o.y() = (int)v.y();
    o.z() = (int)v.z();
    o.w() = (int)v.w();
    reinterpret_cast<sycl::uchar4*>(out)[idx] = o;
  }
}

template<int BLOCKSIZE, int ITEMS_PER_THREAD>
void kernel(const float * __restrict__ A,
            unsigned char *out, const int n,
            const sycl::nd_item<3> &item)
{
  auto g = item.get_group();

  float vals[ITEMS_PER_THREAD];
  unsigned char qvals[ITEMS_PER_THREAD];

  typedef BlockLoad<float, BLOCKSIZE, ITEMS_PER_THREAD> LoadFloat;
  typedef BlockStore<unsigned char, BLOCKSIZE, ITEMS_PER_THREAD> StoreChar;

  auto &loadf_storage = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename LoadFloat::TempStorage>(g);
  auto &storec_storage = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename StoreChar::TempStorage>(g);

  for (size_t i = g.get_group_id(2) * BLOCKSIZE * ITEMS_PER_THREAD;
       i < n; i += g.get_group_range(2) * BLOCKSIZE * ITEMS_PER_THREAD)
  {
      int valid_items = sycl::min(n - i, (size_t)BLOCKSIZE * ITEMS_PER_THREAD);

      // Parameters:
      // block_src_it – [in] The thread block's base iterator for loading from
      // dst_items – [out] Destination to load data into
      // block_items_end – [in] Number of valid items to load
      LoadFloat(loadf_storage, item).Load(&(A[i]), vals, valid_items);

      #pragma unroll
      for(int j = 0; j < ITEMS_PER_THREAD; j++)
          qvals[j] = (int)vals[j];

      StoreChar(storec_storage, item).Store(&(out[i]), qvals, valid_items);
  }
}

template<int BLOCKSIZE, int ITEMS_PER_THREAD>
void kernel2(const float * __restrict__ A,
             unsigned char *out, const int n,
             const sycl::nd_item<3> &item)
{
  auto g = item.get_group();

  float vals[ITEMS_PER_THREAD];
  unsigned char qvals[ITEMS_PER_THREAD];
  auto props = sycl_exp::properties{sycl_exp::data_placement_blocked,
                                    sycl_exp::contiguous_memory,
                                    sycl_exp::full_group
                                   };

  for (size_t i = g.get_group_id(2) * BLOCKSIZE * ITEMS_PER_THREAD;
       i < n; i += g.get_group_range(2) * BLOCKSIZE * ITEMS_PER_THREAD)
  {
    sycl_exp::group_load(g, A+i, sycl::span{vals}, props);

    #pragma unroll
    for(int j = 0; j < ITEMS_PER_THREAD; j++)
        qvals[j] = (int)vals[j];

    sycl_exp::group_store(g, sycl::span{qvals}, out + i, props);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Block access N elements where N is represented as rows x columns\n");
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ncols = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const size_t n = ((size_t)nrows * ncols + 3) / 4 * 4;
  const size_t A_size = n * sizeof(float);
  const size_t out_size = n * sizeof(unsigned char);

  float *A = (float*) malloc (A_size);
  unsigned char *out = (unsigned char*) malloc (out_size);
  unsigned char *out2 = (unsigned char*) malloc (out_size);
  unsigned char *out_ref = (unsigned char*) malloc (out_size);

  std::mt19937 gen{19937};
 
  std::normal_distribution<float> d{-128.0, 127.0};

  for (size_t i = 0; i < n; i++) {
    A[i] = d(gen); 
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_A;
  d_A = (float *)sycl::malloc_device(A_size, q);
  q.memcpy(d_A, A, A_size).wait();

  unsigned char *d_out, *d_out2, *d_out_ref;
  d_out = (unsigned char *)sycl::malloc_device(out_size, q);
  d_out2 = (unsigned char *)sycl::malloc_device(out_size, q);
  d_out_ref = (unsigned char *)sycl::malloc_device(out_size, q);

  const int block_size = 256;
  int cu = q.get_device().get_info<sycl::info::device::max_compute_units>();
  sycl::range<3> gws (1, 1, 16 * cu * block_size);
  sycl::range<3> lws (1, 1, block_size);

  const int items_per_thread = 4;

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(gws, lws),
        [=](sycl::nd_item<3> item) {
          reference(d_A, d_out_ref, n, item);
        });
  });
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(gws, lws),
        [=](sycl::nd_item<3> item) {
          kernel<block_size, items_per_thread>(
              d_A, d_out, n, item);
        });
  });
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(gws, lws),
        [=](sycl::nd_item<3> item) {
          kernel2<block_size, items_per_thread>(
              d_A, d_out2, n, item);
        });
  });

  q.memcpy(out, d_out, out_size);
  q.memcpy(out2, d_out2, out_size);
  q.memcpy(out_ref, d_out_ref, out_size);
  q.wait();

  bool error = false;
  for (size_t i = 0; i < n; i++) {
    unsigned char t = int(A[i]);
    if (out[i] != t) {
      printf("@%zu: out[%u] != %u\n", i, out[i], t);
      error = true;
      break;
    }
    if (out2[i] != t) {
      printf("@%zu: out2[%u] != %u\n", i, out2[i], t);
      error = true;
      break;
    }
    if (out_ref[i] != t) {
      printf("@%zu: out_ref[%u] != %u\n", i, out_ref[i], t);
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(gws, lws),
          [=](sycl::nd_item<3> item) {
            reference(d_A, d_out_ref, n, item);
          });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the reference kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(gws, lws),
          [=](sycl::nd_item<3> item) {
            kernel<block_size, items_per_thread>(
                d_A, d_out, n, item);
          });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the blockAccess kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(gws, lws),
          [=](sycl::nd_item<3> item) {
            kernel2<block_size, items_per_thread>(
                d_A, d_out2, n, item);
          });
    });
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the blockAccess2 kernel: %f (us)\n", (time * 1e-3f) / repeat);

  sycl::free(d_A, q);
  sycl::free(d_out, q);
  sycl::free(d_out2, q);
  sycl::free(d_out_ref, q);
  free(A);
  free(out);
  free(out2);
  free(out_ref);
  return 0;
}
