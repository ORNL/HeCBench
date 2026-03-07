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
#include <cuda.h>
#include "verify.cu"

//----------------------------------------------------------------------------
// Scans each warp in parallel ("warp-scan"), one element per thread.
// uses 2 numElements of shared memory per thread (64 = elements per warp)
//----------------------------------------------------------------------------
#define WARP_SIZE 32
__device__
unsigned int scanwarp(unsigned int val, volatile unsigned int* sData, const int maxlevel)
{
  // The following is the same as 2 * RadixSort::WARP_SIZE * warpId + threadInWarp = 
  // 64*(threadIdx.x >> 5) + (threadIdx.x & (RadixSort::WARP_SIZE - 1))
  int localId = threadIdx.x;
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
__device__
uint4 scan4(const uint4 idata, unsigned int* ptr)
{    

  unsigned int idx = threadIdx.x;

  uint4 val4 = idata;
  unsigned int sum[3];
  sum[0] = val4.x;
  sum[1] = val4.y + sum[0];
  sum[2] = val4.z + sum[1];

  unsigned int val = val4.w + sum[2];

  val = scanwarp(val, ptr, 4);
  __syncthreads();

  if ((idx & (WARP_SIZE - 1)) == WARP_SIZE - 1)
  {
    ptr[idx >> 5] = val + val4.w + sum[2];
  }
  __syncthreads();

  if (idx < WARP_SIZE)
    ptr[idx] = scanwarp(ptr[idx], ptr, 2);

  __syncthreads();

  val += ptr[idx >> 5];

  val4.x = val;
  val4.y = val + sum[0];
  val4.z = val + sum[1];
  val4.w = val + sum[2];

  return val4;
}

__device__
uint4 rank4(const uint4 preds, unsigned int* sMem, unsigned int* numtrue)
{
  int localId = threadIdx.x;
  int localSize = blockDim.x;

  uint4 address = scan4(preds, sMem);

  if (localId == localSize - 1) 
  {
    numtrue[0] = address.w + preds.w;
  }
  __syncthreads();

  uint4 rank;
  int idx = localId*4;
  rank.x = (preds.x) ? address.x : numtrue[0] + idx - address.x;
  rank.y = (preds.y) ? address.y : numtrue[0] + idx + 1 - address.y;
  rank.z = (preds.z) ? address.z : numtrue[0] + idx + 2 - address.z;
  rank.w = (preds.w) ? address.w : numtrue[0] + idx + 3 - address.w;

  return rank;
}

__global__ void radixSortBlocksKeysK(
   unsigned int* keysIn, 
   unsigned int* keysOut,
   const unsigned int nbits,
   const unsigned int startbit)
{
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ unsigned int numtrue[1];
  __shared__ unsigned int sMem[4*128];

  uint4 key = reinterpret_cast<uint4*>(keysIn)[globalId];

  __syncthreads();

  // radixSortBlockKeysOnly(&key, nbits, startbit, sMem, numtrue);
  int localId = threadIdx.x;
  int localSize = blockDim.x;

  for(unsigned int shift = startbit; shift < (startbit + nbits); ++shift)
  {
    uint4 lsb;
    lsb.x = !((key.x >> shift) & 0x1);
    lsb.y = !((key.y >> shift) & 0x1);
    lsb.z = !((key.z >> shift) & 0x1);
    lsb.w = !((key.w >> shift) & 0x1);

    uint4 r;

    r = rank4(lsb, sMem, numtrue);

    // This arithmetic strides the ranks across 4 CTA_SIZE regions
    sMem[(r.x & 3) * localSize + (r.x >> 2)] = key.x;
    sMem[(r.y & 3) * localSize + (r.y >> 2)] = key.y;
    sMem[(r.z & 3) * localSize + (r.z >> 2)] = key.z;
    sMem[(r.w & 3) * localSize + (r.w >> 2)] = key.w;
    __syncthreads();

    // The above allows us to read without 4-way bank conflicts:
    key.x = sMem[localId];
    key.y = sMem[localId +     localSize];
    key.z = sMem[localId + 2 * localSize];
    key.w = sMem[localId + 3 * localSize];

    __syncthreads();
  }

  //keysOut[globalId] = key;
  reinterpret_cast<uint4*>(keysOut)[globalId] = key;  
  
}

  
int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  }
  const int N = atoi(argv[1]);  // assume a multiple of 512
  const int repeat = atoi(argv[2]);
    
  srand(512);
  unsigned int *keys = (unsigned int*) malloc (N * sizeof(unsigned int));
  unsigned int *out = (unsigned int*) malloc (N * sizeof(unsigned int));
  for (int i = 0; i < N; i++)  keys[i] = rand() % 16;

  const unsigned int startbit = 0;
  const unsigned int nbits = 4;
  const unsigned threads = 128; // 1
  const unsigned teams = N/4/threads; // 1

  unsigned int* d_keys;
  cudaMalloc((void**)&d_keys, N*sizeof(unsigned int));
  cudaMemcpy(d_keys, keys, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
  
  unsigned int* d_tempKeys;
  cudaMalloc((void**)&d_tempKeys, N*sizeof(unsigned int));

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    radixSortBlocksKeysK<<<teams, threads>>>(d_keys, d_tempKeys, nbits, startbit);
 
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(out, d_tempKeys, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(d_keys);
  cudaFree(d_tempKeys);

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
