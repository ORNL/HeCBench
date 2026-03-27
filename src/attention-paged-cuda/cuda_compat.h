#pragma once

#ifdef USE_ROCM

#include <hip/hip_runtime.h>
#define GPU_CHECK(x) do { \
    hipError_t err = x; \
    if (err != hipSuccess) { \
        printf("HIP error %s:%d: %s\n", \
               __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(err); \
    } \
} while (0)

#else

#include <cuda_runtime.h>
#define GPU_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", \
               __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(err); \
    } \
} while (0)

#endif

#ifdef USE_ROCM
struct Utils {
  static __host__ int get_warp_size() {
    static bool is_cached = false;
    static int result;

    if (!is_cached) {
      int warp_size;
      GPU_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
      result = warp_size;
      is_cached = true;
    }

    return result;
  }

  static __device__ constexpr int get_warp_size() {
  #ifdef __GFX9__
    return 64;
  #else
    return 32;
  #endif
  }
};

  #define WARP_SIZE Utils::get_warp_size()
#else
  #define WARP_SIZE 32
#endif

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
#else
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor(var, lane_mask, width)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#else
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_DOWN_SYNC(var, lane_delta) \
    __shfl_down_sync(uint32_t(-1), var, lane_delta)
#else
  #define VLLM_SHFL_DOWN_SYNC(var, lane_delta) __shfl_down(var, lane_delta)
#endif

#ifndef USE_ROCM
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    GPU_CHECK(cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL))
#else
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    GPU_CHECK(hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL))
#endif
