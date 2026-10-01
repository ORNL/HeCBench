// Supplemental device-side implementations of SPIR-V sub-group shuffle builtins
// for the AMDGCN target.
//
// This DPC++ install's amdgcn device library (libspirv) does not implement the
// __spirv_GroupNonUniform*Shuffle* family, which oneDPL's work-group scan (used
// by inclusive_scan / sort / etc.) relies on. Without these, device linking
// fails with "undefined symbol: __spirv_GroupNonUniformShuffleUp".
//
// We provide them here as SYCL_EXTERNAL device functions backed by the hardware
// ds_bpermute instruction (a wavefront-wide shuffle). Defining them in a
// separate translation unit (with no call site) avoids the call-resolution
// ambiguity with the compiler's implicit builtin declaration.
#include <sycl/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__

__attribute__((always_inline)) static inline unsigned __amd_lane_id() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

// Read a 32-bit value from lane (self - delta) within the wavefront.
__attribute__((always_inline)) static inline int
__amd_shuffle_up_i32(int v, unsigned delta) {
  int lane = (int)__amd_lane_id();
  int src = lane - (int)delta;
  if (src < 0) src = lane; // underflow is UB in the SYCL contract; stay in range
  return __builtin_amdgcn_ds_bpermute(src << 2, v);
}

__attribute__((always_inline)) static inline long
__amd_shuffle_up_i64(long v, unsigned delta) {
  unsigned lo = (unsigned)(unsigned long)v;
  unsigned hi = (unsigned)((unsigned long)v >> 32);
  lo = (unsigned)__amd_shuffle_up_i32((int)lo, delta);
  hi = (unsigned)__amd_shuffle_up_i32((int)hi, delta);
  return (long)(((unsigned long)hi << 32) | (unsigned long)lo);
}

// Read a 32-bit value from lane (self ^ mask) within the wavefront.
__attribute__((always_inline)) static inline int
__amd_shuffle_xor_i32(int v, unsigned mask) {
  int lane = (int)__amd_lane_id();
  int src = lane ^ (int)mask;
  return __builtin_amdgcn_ds_bpermute(src << 2, v);
}

__attribute__((always_inline)) static inline long
__amd_shuffle_xor_i64(long v, unsigned mask) {
  unsigned lo = (unsigned)(unsigned long)v;
  unsigned hi = (unsigned)((unsigned long)v >> 32);
  lo = (unsigned)__amd_shuffle_xor_i32((int)lo, mask);
  hi = (unsigned)__amd_shuffle_xor_i32((int)hi, mask);
  return (long)(((unsigned long)hi << 32) | (unsigned long)lo);
}

SYCL_EXTERNAL int __spirv_GroupNonUniformShuffleUp(int, int v,
                                                   unsigned delta) noexcept {
  return __amd_shuffle_up_i32(v, delta);
}
SYCL_EXTERNAL long __spirv_GroupNonUniformShuffleUp(int, long v,
                                                    unsigned delta) noexcept {
  return __amd_shuffle_up_i64(v, delta);
}
SYCL_EXTERNAL unsigned long
__spirv_GroupNonUniformShuffleUp(int, unsigned long v, unsigned delta) noexcept {
  return (unsigned long)__amd_shuffle_up_i64((long)v, delta);
}

SYCL_EXTERNAL int __spirv_GroupNonUniformShuffleXor(int, int v,
                                                    unsigned mask) noexcept {
  return __amd_shuffle_xor_i32(v, mask);
}
SYCL_EXTERNAL long __spirv_GroupNonUniformShuffleXor(int, long v,
                                                     unsigned mask) noexcept {
  return __amd_shuffle_xor_i64(v, mask);
}
SYCL_EXTERNAL unsigned long
__spirv_GroupNonUniformShuffleXor(int, unsigned long v, unsigned mask) noexcept {
  return (unsigned long)__amd_shuffle_xor_i64((long)v, mask);
}

// Read a 32-bit value from an absolute lane id within the wavefront.
__attribute__((always_inline)) static inline int
__amd_shuffle_idx_i32(int v, unsigned id) {
  return __builtin_amdgcn_ds_bpermute((int)(id << 2), v);
}

__attribute__((always_inline)) static inline long
__amd_shuffle_idx_i64(long v, unsigned id) {
  unsigned lo = (unsigned)(unsigned long)v;
  unsigned hi = (unsigned)((unsigned long)v >> 32);
  lo = (unsigned)__amd_shuffle_idx_i32((int)lo, id);
  hi = (unsigned)__amd_shuffle_idx_i32((int)hi, id);
  return (long)(((unsigned long)hi << 32) | (unsigned long)lo);
}

// Read a 32-bit value from lane (self + delta) within the wavefront.
__attribute__((always_inline)) static inline int
__amd_shuffle_down_i32(int v, unsigned delta) {
  int lane = (int)__amd_lane_id();
  return __builtin_amdgcn_ds_bpermute((lane + (int)delta) << 2, v);
}

__attribute__((always_inline)) static inline long
__amd_shuffle_down_i64(long v, unsigned delta) {
  unsigned lo = (unsigned)(unsigned long)v;
  unsigned hi = (unsigned)((unsigned long)v >> 32);
  lo = (unsigned)__amd_shuffle_down_i32((int)lo, delta);
  hi = (unsigned)__amd_shuffle_down_i32((int)hi, delta);
  return (long)(((unsigned long)hi << 32) | (unsigned long)lo);
}

SYCL_EXTERNAL int __spirv_GroupNonUniformShuffle(int, int v,
                                                 unsigned id) noexcept {
  return __amd_shuffle_idx_i32(v, id);
}
SYCL_EXTERNAL long __spirv_GroupNonUniformShuffle(int, long v,
                                                  unsigned id) noexcept {
  return __amd_shuffle_idx_i64(v, id);
}
SYCL_EXTERNAL unsigned long
__spirv_GroupNonUniformShuffle(int, unsigned long v, unsigned id) noexcept {
  return (unsigned long)__amd_shuffle_idx_i64((long)v, id);
}

SYCL_EXTERNAL int __spirv_GroupNonUniformShuffleDown(int, int v,
                                                     unsigned delta) noexcept {
  return __amd_shuffle_down_i32(v, delta);
}
SYCL_EXTERNAL long __spirv_GroupNonUniformShuffleDown(int, long v,
                                                      unsigned delta) noexcept {
  return __amd_shuffle_down_i64(v, delta);
}
SYCL_EXTERNAL unsigned long
__spirv_GroupNonUniformShuffleDown(int, unsigned long v,
                                   unsigned delta) noexcept {
  return (unsigned long)__amd_shuffle_down_i64((long)v, delta);
}

#endif // __SYCL_DEVICE_ONLY__
