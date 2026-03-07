//
// An implementation of Parallel Marching Blocks algorithm
//

#include <cstdio>
#include <random>
#include <chrono>
#include <sycl/sycl.hpp>

using uchar4 = sycl::uchar4;
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

template <typename T>
inline T atomicAdd(T *var, T val)
{
  auto atm = sycl::atomic_ref<T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(*var);
  return atm.fetch_add(val);
}

inline float f(unsigned int x, unsigned int y, unsigned int z)
{
  constexpr float d(2.0f / N);
  float xf((int(x - Nd2)) * d);//[-1, 1)
  float yf((int(z - Nd2)) * d);
  float zf((int(z - Nd2)) * d);
  return 1.f - 16.f * xf * yf * zf - 4.f * (xf * xf + yf * yf + zf * zf);
}

inline float zeroPoint(unsigned int x, float v0, float v1, float isoValue)
{
  return ((x * (v1 - isoValue) + (x + 1) * (isoValue - v0)) / (v1 - v0) - Nd2) * (2.0f / N);
}

inline float transformToCoord(unsigned int x)
{
  return (int(x) - int(Nd2)) * (2.0f / N);
}

void computeMinMaxLv1(float*__restrict minMax, float *__restrict sminMax, sycl::nd_item<3> &item)
{
  constexpr unsigned int threadNum(voxelXLv1 * voxelYLv1);
  constexpr unsigned int warpNum(threadNum / 32);
  auto sg = item.get_sub_group();
  int blockIdx_z = item.get_group(0);
  int blockIdx_y = item.get_group(1);
  int blockIdx_x = item.get_group(2);
  int threadIdx_z = item.get_local_id(0);
  int threadIdx_y = item.get_local_id(1);
  int threadIdx_x = item.get_local_id(2);
  unsigned int x(blockIdx_x * (voxelXLv1 - 1) + threadIdx_x);
  unsigned int y(blockIdx_y * (voxelYLv1 - 1) + threadIdx_y);
  unsigned int z(blockIdx_z * (voxelZLv1 - 1));
  unsigned int tid(threadIdx_x + voxelXLv1 * threadIdx_y);
  unsigned int laneid = tid % 32;
  unsigned int blockid(blockIdx_x + gridXLv1 * (blockIdx_y + gridYLv1 * blockIdx_z));
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
    t0 = sycl::shift_group_left(sg, minV, c0);
    t1 = sycl::shift_group_left(sg, maxV, c0);
    if (t0 < minV)minV = t0;
    if (t1 > maxV)maxV = t1;
  }
  if (laneid == 0)
  {
    sminMax[warpid] = minV;
    sminMax[warpid + warpNum] = maxV;
  }
  item.barrier(sycl::access::fence_space::local_space);
  if (warpid == 0)
  {
    minV = sminMax[laneid];
    maxV = sminMax[laneid + warpNum];
#pragma unroll
    for (int c0(warpNum / 2); c0 > 0; c0 /= 2)
    {
      float t0, t1;
      t0 = sycl::shift_group_left(sg, minV, c0);
      t1 = sycl::shift_group_left(sg, maxV, c0);
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

void compactLv1(
  float isoValue,
  const float*__restrict minMax,
  unsigned int*__restrict blockIndices,
  unsigned int*__restrict countedBlockNum,
  unsigned int*__restrict sums,
  sycl::nd_item<1> &item)
{
  auto sg = item.get_sub_group();
  constexpr unsigned int warpNum(countingThreadNumLv1 / 32);
  unsigned int tid(item.get_local_id(0));
  unsigned int laneid = tid % 32;
  unsigned int bIdx(item.get_group(0) * countingThreadNumLv1 + tid);
  unsigned int warpid(tid >> 5);
  unsigned int test;
  if (minMax[2 * bIdx] <= isoValue && minMax[2 * bIdx + 1] >= isoValue)test = 1;
  else test = 0;
  unsigned int testSum(test);
#pragma unroll
  for (int c0(1); c0 < 32; c0 *= 2)
  {
    unsigned int tp(sycl::shift_group_right(sg, testSum, c0));
    if (laneid >= c0) testSum += tp;
  }
  if (laneid == 31)sums[warpid] = testSum;
  item.barrier(sycl::access::fence_space::local_space);
  if (warpid == 0)
  {
    unsigned warpSum = sums[laneid];
#pragma unroll
    for (int c0(1); c0 < warpNum; c0 *= 2)
    {
      unsigned int tp(sycl::shift_group_right(sg, warpSum, c0));
      if (laneid >= c0) warpSum += tp;
    }
    sums[laneid] = warpSum;
  }
  item.barrier(sycl::access::fence_space::local_space);
  if (warpid != 0)testSum += sums[warpid - 1];
  if (tid == countingThreadNumLv1 - 1 && testSum != 0) {
    sums[31] = atomicAdd(countedBlockNum, testSum);
  }
  item.barrier(sycl::access::fence_space::local_space);
  if (test) blockIndices[testSum + sums[31] - 1] = bIdx;
}

void computeMinMaxLv2(
  const unsigned int*__restrict blockIndicesLv1,
  float*__restrict minMax,
  sycl::nd_item<2> &item)
{
  auto sg = item.get_sub_group();
  unsigned int tid(item.get_local_id(1));
  unsigned int voxelOffset(item.get_local_id(0));
  unsigned int blockIdx_x = item.get_group(1);
  unsigned int blockIndex(blockIndicesLv1[blockIdx_x]);
  unsigned int tp(blockIndex);
  unsigned int x((blockIndex % gridXLv1) * (voxelXLv1 - 1) + (voxelOffset % 5) * (voxelXLv2 - 1) + (tid & 3));
  tp /= gridXLv1;
  unsigned int y((tp % gridYLv1) * (voxelYLv1 - 1) + (voxelOffset / 5) * (voxelYLv2 - 1) + (tid >> 2));
  tp /= gridYLv1;
  unsigned int z(tp * (voxelZLv1 - 1));
  float v(f(x, y, z));
  float minV(v), maxV(v);
  unsigned int idx(2 * (voxelOffset + voxelNumLv2 * blockIdx_x));
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
      t0 = sycl::shift_group_left(sg, minV, c1);
      t1 = sycl::shift_group_left(sg, maxV, c1);
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

void compactLv2(
  float isoValue,
  const float*__restrict minMax,
  const unsigned int*__restrict blockIndicesLv1,
  unsigned int*__restrict blockIndicesLv2,
  unsigned int counterBlockNumLv1,
  unsigned int*__restrict countedBlockNumLv2,
  unsigned int*__restrict sums,
  sycl::nd_item<1> &item)
{
  auto sg = item.get_sub_group();
  constexpr unsigned int warpNum(countingThreadNumLv2 / 32);
  unsigned int tid(item.get_local_id(0));
  unsigned int laneid = tid % 32;
  unsigned int warpid(tid >> 5);
  unsigned int id0(tid + item.get_group(0) * countingThreadNumLv2);
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
    unsigned int tp(sycl::shift_group_right(sg, testSum, c0));
    if (laneid >= c0) testSum += tp;
  }
  if (laneid == 31) sums[warpid] = testSum;
  item.barrier(sycl::access::fence_space::local_space);
  if (warpid == 0)
  {
    unsigned int warpSum = sums[laneid];
#pragma unroll
    for (int c0(1); c0 < warpNum; c0 *= 2)
    {
      unsigned int tp(sycl::shift_group_right(sg, warpSum, c0));
      if (laneid >= c0)warpSum += tp;
    }
    sums[laneid] = warpSum;
  }
  item.barrier(sycl::access::fence_space::local_space);
  if (warpid != 0) testSum += sums[warpid - 1];
  if (tid == countingThreadNumLv2 - 1) {
    sums[31] = atomicAdd(countedBlockNumLv2, testSum);
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (test)
  {
    unsigned int bIdx1(blockIndicesLv1[id1]);
    unsigned int bIdx2;
    unsigned int x1, y1, z1;
    unsigned int x2, y2, z2;
    unsigned int tp1(bIdx1);
    unsigned int tp2((tid + item.get_group(0) * countingThreadNumLv2) % voxelNumLv2);
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

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  unsigned int repeat = atoi(argv[1]);

  std::uniform_real_distribution<float>rd(0, 1);
  std::mt19937 mt(123);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *minMaxLv1Device = sycl::malloc_device<float>(blockNum * 2 , q);
  unsigned int *blockIndicesLv1Device = sycl::malloc_device<unsigned int>( blockNum , q);
  unsigned int *countedBlockNumLv1Device = sycl::malloc_device<unsigned int>(1, q);
  unsigned int *countedBlockNumLv2Device = sycl::malloc_device<unsigned int>(1, q);

  unsigned short *distinctEdgesTableDevice = sycl::malloc_device<unsigned short>(256, q);
  q.memcpy(distinctEdgesTableDevice, distinctEdgesTable, sizeof(distinctEdgesTable));

  int *triTableDevice = sycl::malloc_device<int>(256*16, q);
  q.memcpy(triTableDevice, triTable, sizeof(triTable));

  uchar4 *edgeIDTableDevice = sycl::malloc_device<uchar4>(12, q);
  q.memcpy(edgeIDTableDevice, edgeIDTable, sizeof(edgeIDTable));

  unsigned int *countedVerticesNumDevice = sycl::malloc_device<unsigned int>(1, q);
  unsigned int *countedTrianglesNumDevice = sycl::malloc_device<unsigned int>(1, q);

  // simulate rendering without memory allocation for vertices and triangles
  unsigned long long *trianglesDevice = sycl::malloc_device<unsigned long long>(1, q);
  float *coordXDevice = sycl::malloc_device<float>(1, q);
  float *coordYDevice = sycl::malloc_device<float>(1, q);
  float *coordZDevice = sycl::malloc_device<float>(1, q);
  float *coordZPDevice = sycl::malloc_device<float>(1, q);

  const sycl::range<3> BlockSizeLv1{ 1, voxelYLv1, voxelXLv1};
  const sycl::range<3> GridSizeLv1{ gridZLv1, gridYLv1, gridXLv1 };
  const sycl::range<2> BlockSizeLv2{ blockXLv2 * blockYLv2, voxelXLv2 * voxelYLv2 };
  const sycl::range<3> BlockSizeGenerating{ voxelZLv2, voxelYLv2, voxelXLv2 };

  float isoValue(-0.9f);

  unsigned int countedBlockNumLv1;
  unsigned int countedBlockNumLv2;
  unsigned int countedVerticesNum;
  unsigned int countedTrianglesNum;

  float time(0.f);

  for (unsigned int c0(0); c0 < repeat; ++c0)
  {
    q.wait();

    q.memset(countedBlockNumLv1Device, 0, sizeof(unsigned int));
    q.memset(countedBlockNumLv2Device, 0, sizeof(unsigned int));
    q.memset(countedVerticesNumDevice, 0, sizeof(unsigned int));
    q.memset(countedTrianglesNumDevice,0, sizeof(unsigned int));
    q.memset(trianglesDevice, 0, sizeof(unsigned long long));
    q.memset(coordXDevice, 0, sizeof(float));
    q.memset(coordYDevice, 0, sizeof(float));
    q.memset(coordZDevice, 0, sizeof(float));
    q.memset(coordZPDevice, 0, sizeof(float));

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> smem (sycl::range<1>(64), cgh);
      cgh.parallel_for<class min_max1>(
        sycl::nd_range<3>(GridSizeLv1*BlockSizeLv1, BlockSizeLv1), [=] (sycl::nd_item<3> item) {
        computeMinMaxLv1(minMaxLv1Device, smem.get_multi_ptr<sycl::access::decorated::no>().get(), item);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<unsigned int, 1> smem (sycl::range<1>(32), cgh);
      cgh.parallel_for<class compact1>(sycl::nd_range<1>(
        sycl::range<1>(countingBlockNumLv1*countingThreadNumLv1),
        sycl::range<1>(countingThreadNumLv1)), [=] (sycl::nd_item<1> item) {
        compactLv1(isoValue,
                   minMaxLv1Device,
                   blockIndicesLv1Device,
                   countedBlockNumLv1Device,
                   smem.get_multi_ptr<sycl::access::decorated::no>().get(),
                   item);
      });
    });

    q.memcpy(&countedBlockNumLv1, countedBlockNumLv1Device, sizeof(unsigned int));

    float *minMaxLv2Device = sycl::malloc_device<float>(countedBlockNumLv1 * voxelNumLv2 * 2, q);

    const sycl::range<2> GridSizeLv2 (1, countedBlockNumLv1);

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class min_max2>(
        sycl::nd_range<2>(GridSizeLv2*BlockSizeLv2, BlockSizeLv2), [=] (sycl::nd_item<2> item) {
        computeMinMaxLv2(blockIndicesLv1Device, minMaxLv2Device, item);
      });
    });

    unsigned int *blockIndicesLv2Device = sycl::malloc_device<unsigned int>(countedBlockNumLv1 * voxelNumLv2, q);
    unsigned int countingBlockNumLv2((countedBlockNumLv1 * voxelNumLv2 + countingThreadNumLv2 - 1) / countingThreadNumLv2);

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<unsigned int, 1> smem (sycl::range<1>(32), cgh);
      cgh.parallel_for<class compact2>(sycl::nd_range<1>(
        sycl::range<1>(countingBlockNumLv2*countingThreadNumLv2),
        sycl::range<1>(countingThreadNumLv2)), [=] (sycl::nd_item<1> item) {
        compactLv2(isoValue,
                     minMaxLv2Device,
                     blockIndicesLv1Device,
                     blockIndicesLv2Device,
                     countedBlockNumLv1,
                     countedBlockNumLv2Device,
                     smem.get_multi_ptr<sycl::access::decorated::no>().get(),
                     item);
      });
    });

    q.memcpy(&countedBlockNumLv2, countedBlockNumLv2Device, sizeof(unsigned int)).wait();

    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<unsigned short, 3> vertexIndices(sycl::range<3>{voxelZLv2, voxelYLv2, voxelXLv2}, cgh);
      sycl::local_accessor<float, 3> value(sycl::range<3>{voxelZLv2+1, voxelYLv2+1, voxelXLv2+1}, cgh);
      sycl::local_accessor<unsigned int, 1> sumsVertices(sycl::range<1>(32), cgh);
      sycl::local_accessor<unsigned int, 1> sumsTriangles(sycl::range<1>(32), cgh);

      const sycl::range<3> trianglesBlock (1,1,countedBlockNumLv2);
      cgh.parallel_for<class triangles_gen>(
        sycl::nd_range<3>(trianglesBlock*BlockSizeGenerating, BlockSizeGenerating), [=] (sycl::nd_item<3> item) {
        auto sg = item.get_sub_group();
        unsigned int threadIdx_x = item.get_local_id(2);
        unsigned int threadIdx_y = item.get_local_id(1);
        unsigned int threadIdx_z = item.get_local_id(0);

        unsigned int blockId(blockIndicesLv2Device[item.get_group(2)]);
        unsigned int tp(blockId);
        unsigned int x((tp % gridXLv2) * (voxelXLv2 - 1) + threadIdx_x);
        tp /= gridXLv2;
        unsigned int y((tp % gridYLv2) * (voxelYLv2 - 1) + threadIdx_y);
        unsigned int z((tp / gridYLv2) * (voxelZLv2 - 1) + threadIdx_z);
        unsigned int eds(7);
        float v(value[threadIdx_z][threadIdx_y][threadIdx_x] = f(x, y, z));
        if (threadIdx_x == voxelXLv2 - 1)
        {
          eds &= 6;
          value[threadIdx_z][threadIdx_y][voxelXLv2] = f(x + 1, y, z);
          if (threadIdx_y == voxelYLv2 - 1)
            value[threadIdx_z][voxelYLv2][voxelXLv2] = f(x + 1, y + 1, z);
        }
        if (threadIdx_y == voxelYLv2 - 1)
        {
          eds &= 5;
          value[threadIdx_z][voxelYLv2][threadIdx_x] = f(x, y + 1, z);
          if (threadIdx_z == voxelZLv2 - 1)
            value[voxelZLv2][voxelYLv2][threadIdx_x] = f(x, y + 1, z + 1);
        }
        if (threadIdx_z == voxelZLv2 - 1)
        {
          eds &= 3;
          value[voxelZLv2][threadIdx_y][threadIdx_x] = f(x, y, z + 1);
          if (threadIdx_x == voxelXLv2 - 1)
            value[voxelZLv2][threadIdx_y][voxelXLv2] = f(x + 1, y, z + 1);
        }
        eds <<= 13;
        item.barrier(sycl::access::fence_space::local_space);
        unsigned int cubeCase(0);
        if (value[threadIdx_z][threadIdx_y][threadIdx_x] < isoValue) cubeCase |= 1;
        if (value[threadIdx_z][threadIdx_y][threadIdx_x + 1] < isoValue) cubeCase |= 2;
        if (value[threadIdx_z][threadIdx_y + 1][threadIdx_x + 1] < isoValue) cubeCase |= 4;
        if (value[threadIdx_z][threadIdx_y + 1][threadIdx_x] < isoValue) cubeCase |= 8;
        if (value[threadIdx_z + 1][threadIdx_y][threadIdx_x] < isoValue) cubeCase |= 16;
        if (value[threadIdx_z + 1][threadIdx_y][threadIdx_x + 1] < isoValue) cubeCase |= 32;
        if (value[threadIdx_z + 1][threadIdx_y + 1][threadIdx_x + 1] < isoValue) cubeCase |= 64;
        if (value[threadIdx_z + 1][threadIdx_y + 1][threadIdx_x] < isoValue) cubeCase |= 128;

        unsigned int distinctEdges(eds ? distinctEdgesTableDevice[cubeCase] : 0);
        unsigned int numTriangles(eds != 0xe000 ? 0 : distinctEdges & 7);
        unsigned int numVertices(sycl::popcount(distinctEdges &= eds));
        unsigned int laneid = (threadIdx_x + voxelXLv2 * (threadIdx_y + voxelYLv2 * threadIdx_z)) % 32;
        unsigned warpid((threadIdx_x + voxelXLv2 * (threadIdx_y + voxelYLv2 * threadIdx_z)) >> 5);
        constexpr unsigned int threadNum(voxelXLv2 * voxelYLv2 * voxelZLv2);
        constexpr unsigned int warpNum(threadNum / 32);
        unsigned int sumVertices(numVertices);
        unsigned int sumTriangles(numTriangles);

        #pragma unroll
        for (int c0(1); c0 < 32; c0 *= 2)
        {
          unsigned int tp0(sycl::shift_group_right(sg, sumVertices, c0));
          unsigned int tp1(sycl::shift_group_right(sg, sumTriangles, c0));
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
        item.barrier(sycl::access::fence_space::local_space);
        if (warpid == 0)
        {
          unsigned warpSumVertices = sumsVertices[laneid];
          unsigned warpSumTriangles = sumsTriangles[laneid];
          #pragma unroll
          for (int c0(1); c0 < warpNum; c0 *= 2)
          {
            unsigned int tp0(sycl::shift_group_right(sg, warpSumVertices, c0));
            unsigned int tp1(sycl::shift_group_right(sg, warpSumTriangles, c0));
            if (laneid >= c0)
            {
              warpSumVertices += tp0;
              warpSumTriangles += tp1;
            }
          }
          sumsVertices[laneid] = warpSumVertices;
          sumsTriangles[laneid] = warpSumTriangles;
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (warpid != 0)
        {
          sumVertices += sumsVertices[warpid - 1];
          sumTriangles += sumsTriangles[warpid - 1];
        }
        if (eds == 0)
        {
          sumsVertices[31] = atomicAdd(countedVerticesNumDevice, sumVertices);
          sumsTriangles[31] = atomicAdd(countedTrianglesNumDevice, sumTriangles);
        }

        unsigned int interOffsetVertices(sumVertices - numVertices);
        sumVertices = interOffsetVertices + sumsVertices[31];//exclusive offset
        sumTriangles = sumTriangles + sumsTriangles[31] - numTriangles;//exclusive offset
        vertexIndices[threadIdx_z][threadIdx_y][threadIdx_x] = interOffsetVertices | distinctEdges;
        item.barrier(sycl::access::fence_space::local_space);

        for (unsigned int c0(0); c0 < numTriangles; ++c0)
        {
          #pragma unroll
          for (unsigned int c1(0); c1 < 3; ++c1)
          {
            int edgeID(triTableDevice[16 * cubeCase + 3 * c0 + c1]);
            uchar4 edgePos(edgeIDTableDevice[edgeID]);
            unsigned short vertexIndex(
              vertexIndices[threadIdx_z + edgePos.z()][threadIdx_y + edgePos.y()][threadIdx_x + edgePos.x()]);
            unsigned int tp(sycl::popcount(vertexIndex >> (16 - edgePos.w())) + (vertexIndex & 0x1fff));
            atomicAdd(trianglesDevice, (unsigned long long)(sumsVertices[31] + tp));
          }
        }

        // sumVertices may be too large for a GPU memory
        float zp = 0.f, cx = 0.f, cy = 0.f, cz = 0.f;

        if (distinctEdges & (1 << 15))
        {
          zp = zeroPoint(x, v, value[threadIdx_z][threadIdx_y][threadIdx_x + 1], isoValue);
          cy = transformToCoord(y);
          cz = transformToCoord(z);
        }
        if (distinctEdges & (1 << 14))
        {
          cx = transformToCoord(x);
          zp += zeroPoint(y, v, value[threadIdx_z][threadIdx_y + 1][threadIdx_x], isoValue);
          cz += transformToCoord(z);
        }
        if (distinctEdges & (1 << 13))
        {
          cx += transformToCoord(x);
          cy += transformToCoord(y);
          zp += zeroPoint(z, v, value[threadIdx_z + 1][threadIdx_y][threadIdx_x], isoValue);
        }
        atomicAdd(coordXDevice, cx);
        atomicAdd(coordYDevice, cy);
        atomicAdd(coordZDevice, cz);
        atomicAdd(coordZPDevice, zp);
      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time += ktime;

    q.memcpy(&countedVerticesNum, countedVerticesNumDevice, sizeof(unsigned int));
    q.memcpy(&countedTrianglesNum, countedTrianglesNumDevice, sizeof(unsigned int));
    q.wait();

    sycl::free(minMaxLv2Device, q);
    sycl::free(blockIndicesLv2Device, q);
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

  sycl::free(minMaxLv1Device, q);
  sycl::free(blockIndicesLv1Device, q);
  sycl::free(countedBlockNumLv1Device, q);
  sycl::free(countedBlockNumLv2Device, q);
  sycl::free(distinctEdgesTableDevice, q);
  sycl::free(triTableDevice, q);
  sycl::free(edgeIDTableDevice, q);
  sycl::free(countedVerticesNumDevice, q);
  sycl::free(countedTrianglesNumDevice, q);
  sycl::free(trianglesDevice, q);
  sycl::free(coordXDevice, q);
  sycl::free(coordYDevice, q);
  sycl::free(coordZDevice, q);
  sycl::free(coordZPDevice, q);
  if (!ok) return 1;
  return 0;
}
