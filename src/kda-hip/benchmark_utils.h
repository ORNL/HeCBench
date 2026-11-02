#ifndef KDA_BENCHMARK_UTILS_H
#define KDA_BENCHMARK_UTILS_H

#include <hip/hip_bfloat16.h>
#include "host_utils.h"

inline void pack_bf16(const float* src, hip_bfloat16* dst, size_t count)
{
  for (size_t i = 0; i < count; ++i)
    dst[i] = hip_bfloat16(src[i]);
}

inline void unpack_bf16(const hip_bfloat16* src, float* dst, size_t count)
{
  for (size_t i = 0; i < count; ++i)
    dst[i] = float(src[i]);
}

#endif
