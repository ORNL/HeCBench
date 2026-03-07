//
// An implementation of Parallel Marching Blocks algorithm
//

#include <cstdio>
#include <random>
#include <chrono>
#include <cuda.h>
#include "tables.h"
#include <cstdlib>

// problem size
constexpr unsigned int N(1024);
constexpr unsigned int Nd2(N / 2);
constexpr unsigned int voxelXLv1(16);
constexpr unsigned int voxelYLv1(16);
constexpr unsigned int voxelZLv1(64);
constexpr unsigned int gridXLv1((N - 1) / (voxelXLv1 - 1));
constexpr unsigned int gridYLv1((N - 1) / (voxelYLv1 - 1));
constexpr unsigned int gridZLv1((N - 1) / (voxelZLv1 - 1));
constexpr unsigned int countingThreadNumLv1(128);
constexpr unsigned int blockNum(gridXLv1* gridYLv1* gridZLv1);
constexpr unsigned int countingBlockNumLv1(blockNum / countingThreadNumLv1);

constexpr unsigned int voxelXLv2(4);
constexpr unsigned int voxelYLv2(4);
constexpr unsigned int voxelZLv2(8);
constexpr unsigned int blockXLv2(5);
constexpr unsigned int blockYLv2(5);
constexpr unsigned int blockZLv2(9);
constexpr unsigned int voxelNumLv2(blockXLv2* blockYLv2* blockZLv2);

constexpr unsigned int countingThreadNumLv2(1024);
constexpr unsigned int gridXLv2(gridXLv1* blockXLv2);
constexpr unsigned int gridYLv2(gridYLv1* blockYLv2);
//constexpr unsigned int gridZLv2(gridZLv1* blockZLv2);

__inline__ __device__ float f(unsigned int x, unsigned int y, unsigned int z)
{
  constexpr float d(2.0f / N);
  float xf((int(x - Nd2)) * d);//[-1, 1)
  float yf((int(z - Nd2)) * d);
  float zf((int(z - Nd2)) * d);
  return 1.f - 16.f * xf * yf * zf - 4.f * (xf * xf + yf * yf + zf * zf);
}

__inline__ __device__ float zeroPoint(unsigned int x, float v0, float v1, float isoValue)
{
  return ((x * (v1 - isoValue) + (x + 1) * (isoValue - v0)) / (v1 - v0) - Nd2) * (2.0f / N);
}

__inline__ __device__ float transformToCoord(unsigned int x)
{
  return (int(x) - int(Nd2)) * (2.0f / N);
}

__global__ void computeMinMaxLv1(float*__restrict__ minMax)
{
  __shared__ float sminMax[64];
  constexpr unsigned int threadNum(voxelXLv1 * voxelYLv1);
  constexpr unsigned int warpNum(threadNum / 32);
  unsigned int x(blockIdx.x * (voxelXLv1 - 1) + threadIdx.x);
  unsigned int y(blockIdx.y * (voxelYLv1 - 1) + threadIdx.y);
  unsigned int z(blockIdx.z * (voxelZLv1 - 1));
  unsigned int tid(threadIdx.x + voxelXLv1 * threadIdx.y);
  unsigned int laneid = tid % 32;
  unsigned int blockid(blockIdx.x + gridXLv1 * (blockIdx.y + gridYLv1 * blockIdx.z));
  unsigned int warpid(tid >> 5);
  float v(f(x, y, z));
  float minV(v), maxV(v);
  for (int c0(1); c0 < voxelZLv1; ++c0)
  {
    v = f(x, y, z + c0);
    if (v < minV)minV = v;
    if (v > maxV)maxV = v;
  }
#pragma unroll
  for (int c0(16); c0 > 0; c0 /= 2)
  {
    float t0, t1;
    t0 = __shfl_down_sync(0xffffffffu, minV, c0);
    t1 = __shfl_down_sync(0xffffffffu, maxV, c0);
    if (t0 < minV)minV = t0;
    if (t1 > maxV)maxV = t1;
  }
  if (laneid == 0)
  {
    sminMax[warpid] = minV;
    sminMax[warpid + warpNum] = maxV;
  }
  __syncthreads();
  if (warpid == 0)
  {
    minV = sminMax[laneid];
    maxV = sminMax[laneid + warpNum];
#pragma unroll
    for (int c0(warpNum / 2); c0 > 0; c0 /= 2)
    {
      float t0, t1;
      t0 = __shfl_down_sync(0xffffffffu, minV, c0);
      t1 = __shfl_down_sync(0xffffffffu, maxV, c0);
      if (t0 < minV)minV = t0;
      if (t1 > maxV)maxV = t1;
    }
    if (laneid == 0)
    {
      minMax[blockid * 2] = minV;
      minMax[blockid * 2 + 1] = maxV;
    }
  }
}

__global__ void compactLv1(
  float isoValue, 
  const float*__restrict__ minMax,
  unsigned int*__restrict__ blockIndices,
  unsigned int*__restrict__ countedBlockNum)
{
  __shared__ unsigned int sums[32];
  constexpr unsigned int warpNum(countingThreadNumLv1 / 32);
  unsigned int tid(threadIdx.x);
  unsigned int laneid = tid % 32;
  unsigned int bIdx(blockIdx.x * countingThreadNumLv1 + tid);
  unsigned int warpid(tid >> 5);
  unsigned int test;
  if (minMax[2 * bIdx] <= isoValue && minMax[2 * bIdx + 1] >= isoValue)test = 1;
  else test = 0;
  unsigned int testSum(test);
#pragma unroll
  for (int c0(1); c0 < 32; c0 *= 2)
  {
    unsigned int tp(__shfl_up_sync(0xffffffffu, testSum, c0));
    if (laneid >= c0)testSum += tp;
  }
  if (laneid == 31)sums[warpid] = testSum;
  __syncthreads();
  if (warpid == 0)
  {
    unsigned int warpSum = sums[laneid];
#pragma unroll
    for (int c0(1); c0 < warpNum; c0 *= 2)
    {
      unsigned int tp(__shfl_up_sync(0xffffffffu, warpSum, c0));
      if (laneid >= c0) warpSum += tp;
    }
    sums[laneid] = warpSum;
  }
  __syncthreads();
  if (warpid != 0)testSum += sums[warpid - 1];
  if (tid == countingThreadNumLv1 - 1 && testSum != 0)
    sums[31] = atomicAdd(countedBlockNum, testSum);
  __syncthreads();
  if (test)blockIndices[testSum + sums[31] - 1] = bIdx;
}

__global__ void computeMinMaxLv2(
  const unsigned int*__restrict__ blockIndicesLv1,
  float*__restrict__ minMax)
{
  unsigned int tid(threadIdx.x);
  unsigned int voxelOffset(threadIdx.y);
  unsigned int blockIndex(blockIndicesLv1[blockIdx.x]);
  unsigned int tp(blockIndex);
  unsigned int x((blockIndex % gridXLv1) * (voxelXLv1 - 1) + (voxelOffset % 5) * (voxelXLv2 - 1) + (tid & 3));
  tp /= gridXLv1;
  unsigned int y((tp % gridYLv1) * (voxelYLv1 - 1) + (voxelOffset / 5) * (voxelYLv2 - 1) + (tid >> 2));
  tp /= gridYLv1;
  unsigned int z(tp * (voxelZLv1 - 1));
  float v(f(x, y, z));
  float minV(v), maxV(v);
  unsigned int idx(2 * (voxelOffset + voxelNumLv2 * blockIdx.x));
  for (int c0(0); c0 < blockZLv2; ++c0)
  {
    for (int c1(1); c1 < voxelZLv2; ++c1)
    {
      v = f(x, y, z + c1);
      if (v < minV)minV = v;
      if (v > maxV)maxV = v;
    }
    z += voxelZLv2 - 1;
#pragma unroll
    for (int c1(8); c1 > 0; c1 /= 2)
    {
      float t0, t1;
      t0 = __shfl_down_sync(0xffffffffu, minV, c1);
      t1 = __shfl_down_sync(0xffffffffu, maxV, c1);
      if (t0 < minV)minV = t0;
      if (t1 > maxV)maxV = t1;
    }
    if (tid == 0)
    {
      minMax[idx] = minV;
      minMax[idx + 1] = maxV;
      constexpr unsigned int offsetSize(2 * blockXLv2 * blockYLv2);
      idx += offsetSize;
    }
    minV = v;
    maxV = v;
  }
}

__global__ void compactLv2(
  float isoValue,
  const float*__restrict__ minMax,
  const unsigned int*__restrict__ blockIndicesLv1,
  unsigned int*__restrict__ blockIndicesLv2,
  unsigned int counterBlockNumLv1,
  unsigned int*__restrict__ countedBlockNumLv2)
{
  __shared__ unsigned int sums[32];
  constexpr unsigned int warpNum(countingThreadNumLv2 / 32);
  unsigned int tid(threadIdx.x);
  unsigned int laneid = tid % 32;
  unsigned int warpid(tid >> 5);
  unsigned int id0(tid + blockIdx.x * countingThreadNumLv2);
  unsigned int id1(id0 / voxelNumLv2);
  unsigned int test;
  if (id1 < counterBlockNumLv1)
  {
    if (minMax[2 * id0] <= isoValue && minMax[2 * id0 + 1] >= isoValue)
      test = 1;
    else
      test = 0;
  }
  else test = 0;
  unsigned int testSum(test);
#pragma unroll
  for (int c0(1); c0 < 32; c0 *= 2)
  {
    unsigned int tp(__shfl_up_sync(0xffffffffu, testSum, c0));
    if (laneid >= c0)testSum += tp;
  }
  if (laneid == 31)sums[warpid] = testSum;
  __syncthreads();
  if (warpid == 0)
  {
    unsigned warpSum = sums[laneid];
#pragma unroll
    for (int c0(1); c0 < warpNum; c0 *= 2)
    {
      unsigned int tp(__shfl_up_sync(0xffffffffu, warpSum, c0));
      if (laneid >= c0)warpSum += tp;
    }
    sums[laneid] = warpSum;
  }
  __syncthreads();
  if (warpid != 0)testSum += sums[warpid - 1];
  if (tid == countingThreadNumLv2 - 1)
    sums[31] = atomicAdd(countedBlockNumLv2, testSum);
  __syncthreads();

  if (test)
  {
    unsigned int bIdx1(blockIndicesLv1[id1]);
    unsigned int bIdx2;
    unsigned int x1, y1, z1;
    unsigned int x2, y2, z2;
    unsigned int tp1(bIdx1);
    unsigned int tp2((tid + blockIdx.x * countingThreadNumLv2) % voxelNumLv2);
    x1 = tp1 % gridXLv1;
    x2 = tp2 % blockXLv2;
    tp1 /= gridXLv1;
    tp2 /= blockXLv2;
    y1 = tp1 % gridYLv1;
    y2 = tp2 % blockYLv2;
    z1 = tp1 / gridYLv1;
    z2 = tp2 / blockYLv2;
    bIdx2 = x2 + blockXLv2 * (x1 + gridXLv1 * (y2 + blockYLv2 * (y1 + gridYLv1 * (z1 * blockZLv2 + z2))));
    blockIndicesLv2[testSum + sums[31] - 1] = bIdx2;
  }
}

__global__ void generatingTriangles(
  float isoValue, 
  const unsigned int*__restrict__ blockIndicesLv2,
  const unsigned short *__restrict__ distinctEdgesTable,
  const int *__restrict__ triTable,
  const uchar4 *__restrict__ edgeIDTable,
  unsigned int*__restrict__ countedVerticesNum,
  unsigned int*__restrict__ countedTrianglesNum,
  unsigned long long*__restrict__ triangles,
  float*__restrict__ coordX,
  float*__restrict__ coordY,
  float*__restrict__ coordZ,
  float*__restrict__ coordZP)
{
  __shared__ unsigned short vertexIndices[voxelZLv2][voxelYLv2][voxelXLv2];
  __shared__ float value[voxelZLv2 + 1][voxelYLv2 + 1][voxelXLv2 + 1];
  __shared__ unsigned int sumsVertices[32];
  __shared__ unsigned int sumsTriangles[32];

  unsigned int blockId(blockIndicesLv2[blockIdx.x]);
  unsigned int tp(blockId);
  unsigned int x((tp % gridXLv2) * (voxelXLv2 - 1) + threadIdx.x);
  tp /= gridXLv2;
  unsigned int y((tp % gridYLv2) * (voxelYLv2 - 1) + threadIdx.y);
  unsigned int z((tp / gridYLv2) * (voxelZLv2 - 1) + threadIdx.z);
  unsigned int eds(7);
  float v(value[threadIdx.z][threadIdx.y][threadIdx.x] = f(x, y, z));
  if (threadIdx.x == voxelXLv2 - 1)
  {
    eds &= 6;
    value[threadIdx.z][threadIdx.y][voxelXLv2] = f(x + 1, y, z);
    if (threadIdx.y == voxelYLv2 - 1)
      value[threadIdx.z][voxelYLv2][voxelXLv2] = f(x + 1, y + 1, z);
  }
  if (threadIdx.y == voxelYLv2 - 1)
  {
    eds &= 5;
    value[threadIdx.z][voxelYLv2][threadIdx.x] = f(x, y + 1, z);
    if (threadIdx.z == voxelZLv2 - 1)
      value[voxelZLv2][voxelYLv2][threadIdx.x] = f(x, y + 1, z + 1);
  }
  if (threadIdx.z == voxelZLv2 - 1)
  {
    eds &= 3;
    value[voxelZLv2][threadIdx.y][threadIdx.x] = f(x, y, z + 1);
    if (threadIdx.x == voxelXLv2 - 1)
      value[voxelZLv2][threadIdx.y][voxelXLv2] = f(x + 1, y, z + 1);
  }
  eds <<= 13;
  __syncthreads();
  unsigned int cubeCase(0);
  if (value[threadIdx.z][threadIdx.y][threadIdx.x] < isoValue) cubeCase |= 1;
  if (value[threadIdx.z][threadIdx.y][threadIdx.x + 1] < isoValue) cubeCase |= 2;
  if (value[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] < isoValue) cubeCase |= 4;
  if (value[threadIdx.z][threadIdx.y + 1][threadIdx.x] < isoValue) cubeCase |= 8;
  if (value[threadIdx.z + 1][threadIdx.y][threadIdx.x] < isoValue) cubeCase |= 16;
  if (value[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] < isoValue) cubeCase |= 32;
  if (value[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] < isoValue) cubeCase |= 64;
  if (value[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] < isoValue) cubeCase |= 128;

  unsigned int distinctEdges(eds ? distinctEdgesTable[cubeCase] : 0);
  unsigned int numTriangles(eds != 0xe000 ? 0 : distinctEdges & 7);
  unsigned int numVertices(__popc(distinctEdges &= eds));
  unsigned int laneid = (threadIdx.x + voxelXLv2 * (threadIdx.y + voxelYLv2 * threadIdx.z)) % 32;
  unsigned warpid((threadIdx.x + voxelXLv2 * (threadIdx.y + voxelYLv2 * threadIdx.z)) >> 5);
  constexpr unsigned int threadNum(voxelXLv2 * voxelYLv2 * voxelZLv2);
  constexpr unsigned int warpNum(threadNum / 32);
  unsigned int sumVertices(numVertices);
  unsigned int sumTriangles(numTriangles);

#pragma unroll
  for (int c0(1); c0 < 32; c0 *= 2)
  {
    unsigned int tp0(__shfl_up_sync(0xffffffffu, sumVertices, c0));
    unsigned int tp1(__shfl_up_sync(0xffffffffu, sumTriangles, c0));
    if (laneid >= c0)
    {
      sumVertices += tp0;
      sumTriangles += tp1;
    }
  }
  if (laneid == 31)
  {
    sumsVertices[warpid] = sumVertices;
    sumsTriangles[warpid] = sumTriangles;
  }
  __syncthreads();
  if (warpid == 0)
  {
    unsigned warpSumVertices = sumsVertices[laneid];
    unsigned warpSumTriangles = sumsTriangles[laneid];
#pragma unroll
    for (int c0(1); c0 < warpNum; c0 *= 2)
    {
      unsigned int tp0(__shfl_up_sync(0xffffffffu, warpSumVertices, c0));
      unsigned int tp1(__shfl_up_sync(0xffffffffu, warpSumTriangles, c0));
      if (laneid >= c0)
      {
        warpSumVertices += tp0;
        warpSumTriangles += tp1;
      }
    }
    sumsVertices[laneid] = warpSumVertices;
    sumsTriangles[laneid] = warpSumTriangles;
  }
  __syncthreads();
  if (warpid != 0)
  {
    sumVertices += sumsVertices[warpid - 1];
    sumTriangles += sumsTriangles[warpid - 1];
  }
  if (eds == 0)
  {
    sumsVertices[31] = atomicAdd(countedVerticesNum, sumVertices);
    sumsTriangles[31] = atomicAdd(countedTrianglesNum, sumTriangles);
  }

  unsigned int interOffsetVertices(sumVertices - numVertices);
  sumVertices = interOffsetVertices + sumsVertices[31];//exclusive offset
  sumTriangles = sumTriangles + sumsTriangles[31] - numTriangles;//exclusive offset
  vertexIndices[threadIdx.z][threadIdx.y][threadIdx.x] = interOffsetVertices | distinctEdges;
  __syncthreads();

  for (unsigned int c0(0); c0 < numTriangles; ++c0)
  {
#pragma unroll
    for (unsigned int c1(0); c1 < 3; ++c1)
    {
      int edgeID(triTable[16 * cubeCase + 3 * c0 + c1]);
      uchar4 edgePos(edgeIDTable[edgeID]);
      unsigned short vertexIndex(vertexIndices[threadIdx.z + edgePos.z][threadIdx.y + edgePos.y][threadIdx.x + edgePos.x]);
      unsigned int tp(__popc(vertexIndex >> (16 - edgePos.w)) + (vertexIndex & 0x1fff));
      atomicAdd(triangles, (unsigned long long)(sumsVertices[31] + tp));
    }
  }

  // sumVertices may be too large for a GPU memory
  float zp = 0.f, cx = 0.f, cy = 0.f, cz = 0.f;

  if (distinctEdges & (1 << 15))
  {
    zp = zeroPoint(x, v, value[threadIdx.z][threadIdx.y][threadIdx.x + 1], isoValue);
    cy = transformToCoord(y);
    cz = transformToCoord(z);
  }
  if (distinctEdges & (1 << 14))
  {
    cx = transformToCoord(x);
    zp += zeroPoint(y, v, value[threadIdx.z][threadIdx.y + 1][threadIdx.x], isoValue);
    cz += transformToCoord(z);
  }
  if (distinctEdges & (1 << 13))
  {
    cx += transformToCoord(x);
    cy += transformToCoord(y);
    zp += zeroPoint(z, v, value[threadIdx.z + 1][threadIdx.y][threadIdx.x], isoValue);
  }
  atomicAdd(coordX, cx);
  atomicAdd(coordY, cy);
  atomicAdd(coordZ, cz);
  atomicAdd(coordZP, zp);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  unsigned int repeat = atoi(argv[1]);

  std::uniform_real_distribution<float>rd(0, 1);
  std::mt19937 mt(123);

  float* minMaxLv1Device;
  float* minMaxLv2Device;
  unsigned int* blockIndicesLv1Device;
  unsigned int* blockIndicesLv2Device;
  unsigned int* countedBlockNumLv1Device;
  unsigned int* countedBlockNumLv2Device;
  unsigned short* distinctEdgesTableDevice;
  int* triTableDevice;
  uchar4* edgeIDTableDevice;
  unsigned int* countedVerticesNumDevice;
  unsigned int* countedTrianglesNumDevice;
  unsigned long long* trianglesDevice;
  float *coordXDevice;
  float *coordYDevice;
  float *coordZDevice;
  float *coordZPDevice;

  cudaMalloc(&minMaxLv1Device, blockNum * 2 * sizeof(float));
  cudaMalloc(&blockIndicesLv1Device, blockNum * sizeof(unsigned int));
  cudaMalloc(&countedBlockNumLv1Device, sizeof(unsigned int));
  cudaMalloc(&countedBlockNumLv2Device, sizeof(unsigned int));
  cudaMalloc(&distinctEdgesTableDevice, sizeof(distinctEdgesTable));
  cudaMalloc(&triTableDevice, sizeof(triTable));
  cudaMalloc(&edgeIDTableDevice, sizeof(edgeIDTable));
  cudaMalloc(&countedVerticesNumDevice, sizeof(unsigned int));
  cudaMalloc(&countedTrianglesNumDevice, sizeof(unsigned int));
  cudaMemcpy(distinctEdgesTableDevice, distinctEdgesTable, sizeof(distinctEdgesTable), cudaMemcpyHostToDevice);
  cudaMemcpy(triTableDevice, triTable, sizeof(triTable), cudaMemcpyHostToDevice);
  cudaMemcpy(edgeIDTableDevice, edgeIDTable, sizeof(edgeIDTable), cudaMemcpyHostToDevice);

  // simulate rendering without memory allocation for vertices and triangles 
  cudaMalloc(&trianglesDevice, sizeof(unsigned long long));
  cudaMalloc(&coordXDevice, sizeof(float));
  cudaMalloc(&coordYDevice, sizeof(float));
  cudaMalloc(&coordZDevice, sizeof(float));
  cudaMalloc(&coordZPDevice, sizeof(float));

  const dim3 BlockSizeLv1{ voxelXLv1, voxelYLv1, 1 };
  const dim3 GridSizeLv1{ gridXLv1, gridYLv1, gridZLv1 };
  
  const dim3 BlockSizeLv2{ voxelXLv2 * voxelYLv2, blockXLv2 * blockYLv2, 1 };
  const dim3 BlockSizeGenerating{ voxelXLv2, voxelYLv2, voxelZLv2 };

  float isoValue(-0.9f);

  unsigned int countedBlockNumLv1;
  unsigned int countedBlockNumLv2;
  unsigned int countedVerticesNum;
  unsigned int countedTrianglesNum;

  float time(0.f);

  for (unsigned int c0(0); c0 < repeat; ++c0)
  {
    cudaDeviceSynchronize();
    cudaMemset(countedBlockNumLv1Device, 0, sizeof(unsigned int));
    cudaMemset(countedBlockNumLv2Device, 0, sizeof(unsigned int));
    cudaMemset(countedVerticesNumDevice, 0, sizeof(unsigned int));
    cudaMemset(countedTrianglesNumDevice,0, sizeof(unsigned int));
    cudaMemset(trianglesDevice, 0, sizeof(unsigned long long));
    cudaMemset(coordXDevice, 0, sizeof(float));
    cudaMemset(coordYDevice, 0, sizeof(float));
    cudaMemset(coordZDevice, 0, sizeof(float));
    cudaMemset(coordZPDevice, 0, sizeof(float));

    computeMinMaxLv1 <<< GridSizeLv1, BlockSizeLv1 >>> (minMaxLv1Device);
    compactLv1 <<< countingBlockNumLv1, countingThreadNumLv1 >>> (
      isoValue, minMaxLv1Device, blockIndicesLv1Device, countedBlockNumLv1Device);

    cudaMemcpy(&countedBlockNumLv1, countedBlockNumLv1Device, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMalloc(&minMaxLv2Device, countedBlockNumLv1 * voxelNumLv2 * 2 * sizeof(float));

    computeMinMaxLv2 <<< countedBlockNumLv1, BlockSizeLv2 >>> (blockIndicesLv1Device, minMaxLv2Device);

    cudaMalloc(&blockIndicesLv2Device, countedBlockNumLv1 * voxelNumLv2 * sizeof(unsigned int));
    unsigned int countingBlockNumLv2((countedBlockNumLv1 * voxelNumLv2 + countingThreadNumLv2 - 1) / countingThreadNumLv2);

    compactLv2 <<< countingBlockNumLv2, countingThreadNumLv2 >>> (
      isoValue, minMaxLv2Device, blockIndicesLv1Device, blockIndicesLv2Device, countedBlockNumLv1, countedBlockNumLv2Device);

    cudaMemcpy(&countedBlockNumLv2, countedBlockNumLv2Device, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    auto start = std::chrono::steady_clock::now();

    generatingTriangles <<< countedBlockNumLv2, BlockSizeGenerating >>> (
        isoValue, blockIndicesLv2Device,
        distinctEdgesTableDevice, triTableDevice, edgeIDTableDevice,
        countedVerticesNumDevice, countedTrianglesNumDevice, trianglesDevice,
        coordXDevice, coordYDevice, coordZDevice, coordZPDevice);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time += ktime;

    cudaMemcpy(&countedVerticesNum, countedVerticesNumDevice, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&countedTrianglesNum, countedTrianglesNumDevice, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(minMaxLv2Device);
    cudaFree(blockIndicesLv2Device);
  }

  printf("Block Lv1: %u\nBlock Lv2: %u\n", countedBlockNumLv1, countedBlockNumLv2);
  printf("Vertices Size: %u\n", countedBlockNumLv2 * 304);
  printf("Triangles Size: %u\n", countedBlockNumLv2 * 315 * 3);
  printf("Vertices: %u\nTriangles: %u\n", countedVerticesNum, countedTrianglesNum);
  printf("Average kernel execution time (generatingTriangles): %f (s)\n", (time * 1e-9f) / repeat);

  // specific to the problem size
  bool ok = (countedBlockNumLv1 == 8296 && countedBlockNumLv2 == 240380 &&
             countedVerticesNum == 4856560 && countedTrianglesNum == 6101640);
  printf("%s\n", ok ? "PASS" : "FAIL");

  cudaFree(minMaxLv1Device);
  cudaFree(blockIndicesLv1Device);
  cudaFree(countedBlockNumLv1Device);
  cudaFree(countedBlockNumLv2Device);
  cudaFree(distinctEdgesTableDevice);
  cudaFree(triTableDevice);
  cudaFree(edgeIDTableDevice);
  cudaFree(countedVerticesNumDevice);
  cudaFree(countedTrianglesNumDevice);
  cudaFree(trianglesDevice);
  cudaFree(coordXDevice);
  cudaFree(coordYDevice);
  cudaFree(coordZDevice);
  cudaFree(coordZPDevice);
  if (!ok) return 1;
  return 0;
}
