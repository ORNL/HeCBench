/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "verify.cpp"

using uint4 = sycl::uint4;

//----------------------------------------------------------------------------
// Scans each warp in parallel ("warp-scan"), one element per thread.
// uses 2 numElements of shared memory per thread (64 = elements per warp)
//----------------------------------------------------------------------------
#define WARP_SIZE 32

uint scanwarp(sycl::nd_item<1> &item, uint val, volatile uint* sData, const int maxlevel)
{
  // The following is the same as 2 * RadixSort::WARP_SIZE * warpId + threadInWarp =
  // 64*(threadIdx.x >> 5) + (threadIdx.x & (RadixSort::WARP_SIZE - 1))
  int localId = item.get_local_id(0);
  int idx = 2 * localId - (localId & (WARP_SIZE - 1));
  sData[idx] = 0;
  idx += WARP_SIZE;
  sData[idx] = val;

  if (0 <= maxlevel) { sData[idx] += sData[idx - 1]; }
  if (1 <= maxlevel) { sData[idx] += sData[idx - 2]; }
  if (2 <= maxlevel) { sData[idx] += sData[idx - 4]; }
  if (3 <= maxlevel) { sData[idx] += sData[idx - 8]; }
  if (4 <= maxlevel) { sData[idx] += sData[idx -16]; }

  return sData[idx] - val;  // convert inclusive -> exclusive
}

//----------------------------------------------------------------------------
// scan4 scans 4*RadixSort::CTA_SIZE numElements in a block (4 per thread), using
// a warp-scan algorithm
//----------------------------------------------------------------------------
uint4 scan4(sycl::nd_item<1> item, const uint4 idata, uint* ptr)
{

  uint idx = item.get_local_id(0);

  uint4 val4 = idata;
  uint sum[3];
  sum[0] = val4.x();
  sum[1] = val4.y() + sum[0];
  sum[2] = val4.z() + sum[1];

  uint val = val4.w() + sum[2];

  val = scanwarp(item, val, ptr, 4);
  item.barrier(sycl::access::fence_space::local_space);

  if ((idx & (WARP_SIZE - 1)) == WARP_SIZE - 1)
  {
    ptr[idx >> 5] = val + val4.w() + sum[2];
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (idx < WARP_SIZE)
    ptr[idx] = scanwarp(item, ptr[idx], ptr, 2);

  item.barrier(sycl::access::fence_space::local_space);

  val += ptr[idx >> 5];

  val4.x() = val;
  val4.y() = val + sum[0];
  val4.z() = val + sum[1];
  val4.w() = val + sum[2];

  return val4;
}

uint4 rank4(sycl::nd_item<1> &item, const uint4 preds, uint* sMem, uint &numtrue)
{
  int localId = item.get_local_id(0);
  int localSize = item.get_local_range(0);

  uint4 address = scan4(item, preds, sMem);

  if (localId == localSize - 1)
  {
    numtrue = address.w() + preds.w();
  }
  item.barrier(sycl::access::fence_space::local_space);

  uint4 rank;
  int idx = localId*4;
  rank.x() = (preds.x()) ? address.x() : numtrue + idx - address.x();
  rank.y() = (preds.y()) ? address.y() : numtrue + idx + 1 - address.y();
  rank.z() = (preds.z()) ? address.z() : numtrue + idx + 2 - address.z();
  rank.w() = (preds.w()) ? address.w() : numtrue + idx + 3 - address.w();

  return rank;
}

void radixSortBlocksKeysK(
    sycl::nd_item<1> &item,
    uint*__restrict keysIn,
    uint*__restrict keysOut,
    const uint nbits,
    const uint startbit,
    uint*__restrict sMem,
    uint &numtrue)
{
  int globalId = item.get_global_id(0);
  int localId = item.get_local_id(0);
  int localSize = item.get_local_range(0);

  uint4 key = reinterpret_cast<uint4*>(keysIn)[globalId];

  item.barrier(sycl::access::fence_space::local_space);

  for(uint shift = startbit; shift < (startbit + nbits); ++shift)
  {
    uint4 lsb;
    lsb.x() = !((key.x() >> shift) & 0x1);
    lsb.y() = !((key.y() >> shift) & 0x1);
    lsb.z() = !((key.z() >> shift) & 0x1);
    lsb.w() = !((key.w() >> shift) & 0x1);

    uint4 r;

    r = rank4(item, lsb, sMem, numtrue);

    // This arithmetic strides the ranks across 4 CTA_SIZE regions
    sMem[(r.x() & 3) * localSize + (r.x() >> 2)] = key.x();
    sMem[(r.y() & 3) * localSize + (r.y() >> 2)] = key.y();
    sMem[(r.z() & 3) * localSize + (r.z() >> 2)] = key.z();
    sMem[(r.w() & 3) * localSize + (r.w() >> 2)] = key.w();
    item.barrier(sycl::access::fence_space::local_space);

    // The above allows us to read without 4-way bank conflicts:
    key.x() = sMem[localId];
    key.y() = sMem[localId +     localSize];
    key.z() = sMem[localId + 2 * localSize];
    key.w() = sMem[localId + 3 * localSize];

    item.barrier(sycl::access::fence_space::local_space);
  }

  reinterpret_cast<uint4*>(keysOut)[globalId] = key;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  }
  const int N = atoi(argv[1]);  // assume a multiple of 512
  const int repeat = atoi(argv[2]);

  srand(512);
  uint *keys = (uint*) malloc (N * sizeof(uint));
  uint *out = (uint*) malloc (N * sizeof(uint));
  for (int i = 0; i < N; i++)  keys[i] = rand() % 16;

  const uint startbit = 0;
  const uint nbits = 4;
  const unsigned threads = 128;
  const unsigned teams = N/4/threads;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  uint *d_keys = sycl::malloc_device<uint>(N, q);
  q.memcpy(d_keys, keys, sizeof(uint) * N);

  uint *d_tempKeys = sycl::malloc_device<uint>(N, q);

  sycl::range<1> gws (teams * threads);
  sycl::range<1> lws (threads);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<uint, 1> sMem(sycl::range<1>(4*threads), cgh);
      sycl::local_accessor<uint, 0> numtrue(cgh);
      cgh.parallel_for<class radixSort_blocksKeys>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        radixSortBlocksKeysK(item, d_keys, d_tempKeys,
                             nbits, startbit,
                             sMem.get_multi_ptr<sycl::access::decorated::no>().get(),
                             numtrue);
      });
    });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(out, d_tempKeys, sizeof(uint) * N).wait();
  sycl::free(d_tempKeys, q);
  sycl::free(d_keys, q);

  bool check = verify(out, keys, threads, N);
  if (check)
    printf("PASS\n");
  else
    printf("FAIL\n");

  free(keys);
  free(out);
  if (!check) return 1;
  return 0;
}
