#pragma once

// One shared-memory choice for the CUDA, HIP, and SYCL fused kernels.
//
//   device >= 82432 bytes: ld 129, 64x64 scratch next to the factor
//   device >= 65536 bytes: ld 128, 64x64 scratch in global memory
//   otherwise rejected
// A larger ceiling does not add another buffer.

#include <cstddef>
#include <cstdio>

constexpr size_t kVdiTightBytes = 128ull * 128 * sizeof(float);
constexpr size_t kVdiPaddedBytes = (129ull * 128 + 64ull * 64) * sizeof(float);

struct VdiLocalLayout {
  int ld;
  int scratch_local;
  size_t local_bytes;
};

inline bool vdi_bind_local(size_t device_max, VdiLocalLayout& out) {
  if (device_max >= kVdiPaddedBytes) {
    out.ld = 129;
    out.local_bytes = kVdiPaddedBytes;
    out.scratch_local = 1;
  } else if (device_max >= kVdiTightBytes) {
    out.ld = 128;
    out.local_bytes = kVdiTightBytes;
    out.scratch_local = 0;
  } else {
    fprintf(stderr, "fused: device local memory %zu is below %zu\n", device_max,
            kVdiTightBytes);
    return false;
  }
  return true;
}

inline void vdi_format_layout(char* buf, size_t n, const VdiLocalLayout& lay,
                              size_t device_max) {
  snprintf(buf, n, "ld %d, local %zu of %zu bytes, %s", lay.ld, lay.local_bytes,
           device_max,
           lay.scratch_local ? "64x64 scratch in local memory"
                             : "64x64 scratch in global memory");
}
