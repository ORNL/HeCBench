#include <chrono>
#include <random>
#include <cstdio>
#include <cmath>
#include <sycl/sycl.hpp>
#include "kernels.h"
#include "utils.h"

template <typename Td, typename Ts>
void convert(sycl::queue &q, bool isE4M3, int nelems, int niters)
{
  Ts *h_src = (Ts*) malloc (nelems * sizeof(Ts));
  Td *h_dst = (Td*) malloc (nelems * sizeof(Td));
  Td *r_dst = (Td*) malloc (nelems * sizeof(Td));

  init(isE4M3, h_src, nelems); 

  Ts *src = sycl::malloc_device<Ts>(nelems, q);
  q.memcpy(src, h_src, nelems * sizeof(Ts));

  Td *dst = sycl::malloc_device<Td>(nelems, q);

  const int block_size = 256;
  const int num_blocks = (nelems + block_size - 1) / block_size;

  sycl::range<1> gws (num_blocks * block_size);
  sycl::range<1> lws (block_size);

  // Warm-up run
  for (int i = 0; i < 30; i++) {
    if (isE4M3) {
      q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        ref_fp32_cvt_e4m3<Td, Ts>(dst, src, nelems, item);
      });
      q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        fp32_cvt_e4m3<Td, Ts>(dst, src, nelems, item);
      });
    }
    else {
      q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        ref_fp32_cvt_e5m2<Td, Ts>(dst, src, nelems, item);
      });
      q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        fp32_cvt_e5m2<Td, Ts>(dst, src, nelems, item);
      });
    }
  }

  q.wait();

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    if (isE4M3)
      q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        ref_fp32_cvt_e4m3<Td, Ts>(dst, src, nelems, item);
      });
    else
      q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        ref_fp32_cvt_e5m2<Td, Ts>(dst, src, nelems, item);
      });
  }
  q.wait();
  auto end = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>
                (end - start).count() * 1.0 / niters;
  double size = (sizeof(Td) + sizeof(Ts)) * nelems;
  printf("size(GB):%.2f, average time of reference (sec):%f, BW:%f\n", size, time * 1e-9, size / time);

  q.memcpy(r_dst, dst, nelems * sizeof(Td)).wait();

#ifdef DEBUG
  printf("Print the first 10 hex values:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f -> %x\n", h_src[i], r_dst[i]);
  }
#endif

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    if (isE4M3)
      q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        fp32_cvt_e4m3<Td, Ts>(dst, src, nelems, item);
      });
    else
      q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        fp32_cvt_e5m2<Td, Ts>(dst, src, nelems, item);
      });
  }
  q.wait();
  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>
                (end - start).count() * 1.0 / niters;
  size = (sizeof(Td) + sizeof(Ts)) * nelems;
  printf("size(GB):%.2f, average time of optimized (sec):%f, BW:%f\n", size, time * 1e-9, size / time);

  q.memcpy(h_dst, dst, nelems * sizeof(Td)).wait();

#ifdef DEBUG
  printf("Print the first 10 hex values:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f -> %x\n", h_src[i], h_dst[i]);
  }
#endif

  bool ok = true;
  for (int i = 0; i < nelems; i++) {
    if (abs(int8_t(h_dst[i] - r_dst[i])) > 1) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(src, q);
  sycl::free(dst, q);
  free(h_src);
  free(h_dst);
  free(r_dst);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int niters = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("float -> fp8 E4M3\n");
  convert<uint8_t, float>(q, true, nelems, niters); 

  printf("float -> fp8 E5M2\n");
  convert<uint8_t, float>(q, false, nelems, niters); 

  return 0;
}
