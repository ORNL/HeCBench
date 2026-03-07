// *********************************************************************
// A simple demo application that implements a
// vector dot product computation in INT8 between two arrays
// *********************************************************************

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <random>
#include <type_traits>
#include <cuda.h>
#include <cub/cub.cuh>

inline void GPU_CHECK(cudaError_t status) {
  if (status != cudaSuccess) {
    printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
    throw std::logic_error("cuda API failed");
  }
}

size_t shrRoundUp(int group_size, size_t global_size) 
{
  if (global_size == 0) return group_size;
  int r = global_size % group_size;
  return (r == 0) ? global_size : global_size + group_size - r;
}

template <typename T, int M>
__global__
void dot_product(const T *__restrict__ a,
                 const T *__restrict__ b,
                       T *__restrict__ d,
                 const size_t n)
{
  T sum = 0;
  for(size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < n; idx += gridDim.x * blockDim.x) {
    size_t iInOffset = idx * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      if constexpr (std::is_same_v<T, unsigned>) {
        const uint8_t * a8 = (const uint8_t *) &a[iInOffset + i];
        const uint8_t * b8 = (const uint8_t *) &b[iInOffset + i];
        for (int k = 0; k < M; k++) 
          sum += a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
      }
      else {
        const int8_t * a8 = (const int8_t *) &a[iInOffset + i];
        const int8_t * b8 = (const int8_t *) &b[iInOffset + i];
        for (int k = 0; k < M; k++) 
          sum += a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
      }
    }
  }

  using BlockReduce = cub::BlockReduce<T, 1024>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T aggregate = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0)
    d[blockIdx.x] = aggregate;
}

template <typename T, int M>
__global__
void dot_product2(const T *__restrict__ a,
                  const T *__restrict__ b,
                        T *__restrict__ d,
                  const size_t n)
{
  T sum = 0;
  for(size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < n; idx += gridDim.x * blockDim.x) {
    size_t iInOffset = idx * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      for (int k = 0; k < M; k++) 
        sum = __dp4a(a[iInOffset + i], b[iInOffset + i], sum);
    }
  }

  using BlockReduce = cub::BlockReduce<T, 1024>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T aggregate = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0)
    d[blockIdx.x] = aggregate;
}

template <typename T>
void dot (const size_t iNumElements, const int iNumIterations)
{
  // set and log Global and Local work size dimensions
  int szLocalWorkSize = 1024;
  // rounded up to the nearest multiple of the LocalWorkSize
  size_t szGlobalWorkSize = shrRoundUp(szLocalWorkSize, iNumElements);

  printf("Global Work Size \t\t= %zu\nLocal Work Size \t\t= %d\n",
         szGlobalWorkSize, szLocalWorkSize);

  const size_t src_size = szGlobalWorkSize;
  const size_t src_size_bytes = src_size * sizeof(T);

  const size_t grid_size = shrRoundUp(1, szGlobalWorkSize / (szLocalWorkSize * 4));

  // Allocate and initialize host arrays
  T* srcA = (T*) malloc (src_size_bytes);
  T* srcB = (T*) malloc (src_size_bytes);
  T*  dst = (T*) malloc (grid_size * sizeof(T));

  size_t i;
  std::mt19937 engine(19937);
  std::uniform_int_distribution<T> dis (0, 255);

  T dst_ref = 0;
  for (i = 0; i < iNumElements; ++i)
  {
    T s[4];
    for (int k = 0; k < 4; k++) s[k] = dis(engine);
    srcB[i] = srcA[i] = s[0] | (s[1] << 8) | (s[2] << 16) | (s[3] << 24);
    if constexpr (std::is_same_v<T, int>) {
      for (int k = 0; k < 4; k++) if (s[k] >= 128) s[k] -= 256;
    }
    for (int k = 0; k < 4; k++) dst_ref += s[k] * s[k];
  }
  for (i = iNumElements; i < src_size; ++i) srcA[i] = srcB[i] = 0;

  T *d_srcA, *d_srcB, *d_dst;

  GPU_CHECK(cudaMalloc((void**)&d_srcA, src_size_bytes));
  GPU_CHECK(cudaMemcpy(d_srcA, srcA, src_size_bytes, cudaMemcpyHostToDevice));

  GPU_CHECK(cudaMalloc((void**)&d_srcB, src_size_bytes));
  GPU_CHECK(cudaMemcpy(d_srcB, srcB, src_size_bytes, cudaMemcpyHostToDevice));

  GPU_CHECK(cudaMalloc((void**)&d_dst, grid_size * sizeof(T)));

  dim3 grid (grid_size);
  dim3 block (szLocalWorkSize);

  const int M = 1; // multiplier of dp4a operations per thread in the kernels

  // warmup
  for (i = 0; i < 100; i++) {
    dot_product<T, M><<<grid, block>>>(d_srcA, d_srcB, d_dst, src_size / 4);
    dot_product2<T, M><<<grid, block>>>(d_srcA, d_srcB, d_dst, src_size / 4);
  }

  GPU_CHECK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    dot_product<T, M><<<grid, block>>>(d_srcA, d_srcB, d_dst, src_size / 4);
  }

  GPU_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);

  GPU_CHECK(cudaMemcpy(dst, d_dst, grid_size * sizeof(T), cudaMemcpyDeviceToHost));
  T dst_dev = 0;
  for (i = 0; i < grid_size; i++) dst_dev += dst[i];
  printf("%s\n\n", dst_dev == M * dst_ref ? "PASS" : "FAIL");
  if (dst_dev != M * dst_ref) exit(1);

  start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    dot_product2<T, M><<<grid, block>>>(d_srcA, d_srcB, d_dst, src_size / 4);
  }

  GPU_CHECK(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);

  GPU_CHECK(cudaMemcpy(dst, d_dst, grid_size * sizeof(T), cudaMemcpyDeviceToHost));
  dst_dev = 0;
  for (i = 0; i < grid_size; i++) dst_dev += dst[i];
  printf("%s\n\n", dst_dev == M * dst_ref ? "PASS" : "FAIL");
  if (dst_dev != M * dst_ref) exit(1);

  GPU_CHECK(cudaFree(d_dst));
  GPU_CHECK(cudaFree(d_srcA));
  GPU_CHECK(cudaFree(d_srcB));

  free(srcA);
  free(srcB);
  free(dst);
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t iNumElements = atol(argv[1]);
  const int iNumIterations = atoi(argv[2]);

  printf("------------- Data type is int32 ---------------\n");
  dot<int>(iNumElements, iNumIterations);
  printf("------------- Data type is uint32 ---------------\n");
  dot<unsigned>(iNumElements, iNumIterations);

  return EXIT_SUCCESS;
}
