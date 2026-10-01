#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define  bidx  item.get_group(0)
#define  tidx  item.get_local_id(0)

#include "reference.h"

void intt_3_64k_modcrt(
        sycl::nd_item<1> &item,
        uint64 *__restrict buffer,
        uint32 *__restrict dst,
  const uint64 *__restrict src)
{
  register uint64 samples[8], s8[8];
  register uint32 fmem, tmem, fbuf, tbuf;
  fmem = (bidx<<9)|((tidx&0x3E)<<3)|(tidx&0x1);
  tbuf = tidx<<3;
  fbuf = ((tidx&0x38)<<3) | (tidx&0x7);
  tmem = (bidx<<9)|((tidx&0x38)<<3) | (tidx&0x7);
#pragma unroll
  for (int i=0; i<8; i++)
    samples[i] = src[fmem|(i<<1)];
  ntt8(samples);

#pragma unroll
  for (int i=0; i<8; i++)
    buffer[tbuf|i] = _ls_modP(samples[i], ((tidx&0x1)<<2)*i*3);
  item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
  for (int i=0; i<8; i++)
    samples[i] = buffer[fbuf|(i<<3)];

#pragma unroll
  for (int i=0; i<4; i++) {
    s8[2*i] = _add_modP(samples[2*i], samples[2*i+1]);
    s8[2*i+1] = _sub_modP(samples[2*i], samples[2*i+1]);
  }

#pragma unroll
  for (int i=0; i<8; i++) {
    dst[(((tmem|(i<<3))&0xf)<<12)|((tmem|(i<<3))>>4)] =
      (uint32)(_mul_modP(s8[i], 18446462594437939201UL, valP));
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int nttLen = 64 * 1024;
  const int blockSize = 512;
  const int threadsPerBlock = 64;
  uint64 *ntt = (uint64*) malloc (nttLen*sizeof(uint64));
  uint32 *res = (uint32*) malloc (nttLen*sizeof(uint32));
  uint32 *ref = (uint32*) malloc (nttLen*sizeof(uint32));

  srand(123);
  for (int i = 0; i < nttLen; i++) {
    uint64 hi = rand();
    uint64 lo = rand();
    ntt[i] = (hi << 32) | lo;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  uint64 *d_ntt = sycl::malloc_device<uint64>(nttLen, q);
  uint32 *d_res = sycl::malloc_device<uint32>(nttLen, q);
  q.memcpy(d_ntt, ntt, nttLen*sizeof(uint64));

  sycl::range<1> gws (nttLen/blockSize * threadsPerBlock);
  sycl::range<1> lws (threadsPerBlock);

  auto launch = [&]() {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<uint64, 1> sm (sycl::range<1>(512), cgh);
      cgh.parallel_for<class intt_modcrt>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        intt_3_64k_modcrt(item, sm.get_multi_ptr<sycl::access::decorated::no>().get(), d_res, d_ntt);
      });
    });
  };

  reference(ref, ntt, nttLen, blockSize, threadsPerBlock);
  launch();
  q.memcpy(res, d_res, nttLen*sizeof(uint32)).wait();

  for (int i = 0; i < nttLen; i++) {
    if (res[i] != ref[i]) {
      //printf("Mismatch at index %d: device=%u, reference=%u\n", i, res[i], ref[i]);
      printf("FAIL\n");
      sycl::free(d_ntt, q);
      sycl::free(d_res, q);
      free(ntt);
      free(res);
      free(ref);
      return 1;
    }
  }
  printf("PASS\n");

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    launch();

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  sycl::free(d_ntt, q);
  sycl::free(d_res, q);
  free(ntt);
  free(res);
  free(ref);
  return 0;
}
