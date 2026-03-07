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
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

inline void GPU_CHECK(hipError_t status) {
  if (status != hipSuccess) {
    printf("hip API failed with status %d: %s\n", status, hipGetErrorString(status));
    throw std::logic_error("hip API failed");
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

  using BlockReduce = hipcub::BlockReduce<T, 1024>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T aggregate = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0)
    d[blockIdx.x] = aggregate;
}

template <typename T>
__device__ __forceinline__
T __dp4a(const T a, const T b, T c) {
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
  if constexpr (std::is_same_v<T, unsigned>)
    c = __builtin_amdgcn_udot4(a, b, c, false);
  else
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3) || defined(RDNA4)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(RDNA1) || defined(__gfx900__)
  if constexpr (std::is_same_v<T, unsigned>) {
    unsigned tmp1;
    unsigned tmp2;
    asm("\n \
        v_mul_i32_i24 %1, %3, %4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, %3, %4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, %3, %4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, %3, %4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
  } else {
    int tmp1;
    int tmp2;
    asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
  }
#endif
  return c;
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

  using BlockReduce = hipcub::BlockReduce<T, 1024>;
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

  GPU_CHECK(hipMalloc((void**)&d_srcA, src_size_bytes));
  GPU_CHECK(hipMemcpy(d_srcA, srcA, src_size_bytes, hipMemcpyHostToDevice));

  GPU_CHECK(hipMalloc((void**)&d_srcB, src_size_bytes));
  GPU_CHECK(hipMemcpy(d_srcB, srcB, src_size_bytes, hipMemcpyHostToDevice));

  GPU_CHECK(hipMalloc((void**)&d_dst, grid_size * sizeof(T)));

  dim3 grid (grid_size);
  dim3 block (szLocalWorkSize);

  const int M = 1; // multiplier of dp4a operations per thread in the kernels

  // warmup
  for (i = 0; i < 100; i++) {
    dot_product<T, M><<<grid, block>>>(d_srcA, d_srcB, d_dst, src_size / 4);
    dot_product2<T, M><<<grid, block>>>(d_srcA, d_srcB, d_dst, src_size / 4);
  }

  GPU_CHECK(hipDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    dot_product<T, M><<<grid, block>>>(d_srcA, d_srcB, d_dst, src_size / 4);
  }

  GPU_CHECK(hipDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);

  GPU_CHECK(hipMemcpy(dst, d_dst, grid_size * sizeof(T), hipMemcpyDeviceToHost));
  T dst_dev = 0;
  for (i = 0; i < grid_size; i++) dst_dev += dst[i];
  printf("%s\n\n", dst_dev == M * dst_ref ? "PASS" : "FAIL");
  if (dst_dev != M * dst_ref) exit(1);

  start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    dot_product2<T, M><<<grid, block>>>(d_srcA, d_srcB, d_dst, src_size / 4);
  }

  GPU_CHECK(hipDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);

  GPU_CHECK(hipMemcpy(dst, d_dst, grid_size * sizeof(T), hipMemcpyDeviceToHost));
  dst_dev = 0;
  for (i = 0; i < grid_size; i++) dst_dev += dst[i];
  printf("%s\n\n", dst_dev == M * dst_ref ? "PASS" : "FAIL");
  if (dst_dev != M * dst_ref) exit(1);

  GPU_CHECK(hipFree(d_dst));
  GPU_CHECK(hipFree(d_srcA));
  GPU_CHECK(hipFree(d_srcB));

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
