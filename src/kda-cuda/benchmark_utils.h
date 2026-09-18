#ifndef KDA_BENCHMARK_UTILS_H
#define KDA_BENCHMARK_UTILS_H

#include <cuda_bf16.h>
#include "host_utils.h"

inline void pack_bf16(const float* src, nv_bfloat16* dst, size_t count)
{
  for (size_t i = 0; i < count; ++i)
    dst[i] = __float2bfloat16(src[i]);
}

inline void unpack_bf16(const nv_bfloat16* src, float* dst, size_t count)
{
  for (size_t i = 0; i < count; ++i)
    dst[i] = __bfloat162float(src[i]);
}

#endif
