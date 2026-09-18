#ifndef KDA_BENCHMARK_UTILS_H
#define KDA_BENCHMARK_UTILS_H

#include <sycl/sycl.hpp>
#include "host_utils.h"

using kda_bf16 = sycl::ext::oneapi::bfloat16;

inline void pack_bf16(const float* src, kda_bf16* dst, size_t count)
{
  for (size_t i = 0; i < count; ++i)
    dst[i] = kda_bf16(src[i]);
}

inline void unpack_bf16(const kda_bf16* src, float* dst, size_t count)
{
  for (size_t i = 0; i < count; ++i)
    dst[i] = float(src[i]);
}

#endif
