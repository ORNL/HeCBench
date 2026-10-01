#ifndef _PinnedVector_hpp_
#define _PinnedVector_hpp_

// A std::allocator drop-in that backs storage with CUDA page-locked (pinned)
// host memory. Data placed here is transferred to the device at full PCIe
// bandwidth instead of the reduced rate the driver achieves when it has to
// stage pageable memory through an internal pinned buffer. Used for the large
// matrix (row_offsets/packed_cols/packed_coefs) and vector coefficient arrays
// that are copied host->device once before the CG solve.

#include <cstddef>
#include <new>
#include <vector>
#include <cuda_runtime.h>

namespace miniFE {

template <typename T>
struct PinnedAllocator {
  using value_type = T;

  PinnedAllocator() noexcept = default;
  template <typename U>
  PinnedAllocator(const PinnedAllocator<U> &) noexcept {}

  T *allocate(std::size_t n) {
    if (n == 0)
      return nullptr;
    void *p = nullptr;
    if (cudaMallocHost(&p, n * sizeof(T)) != cudaSuccess || p == nullptr)
      throw std::bad_alloc();
    return static_cast<T *>(p);
  }

  void deallocate(T *p, std::size_t) noexcept {
    if (p)
      cudaFreeHost(p);
  }
};

template <typename T, typename U>
bool operator==(const PinnedAllocator<T> &, const PinnedAllocator<U> &) noexcept {
  return true;
}
template <typename T, typename U>
bool operator!=(const PinnedAllocator<T> &, const PinnedAllocator<U> &) noexcept {
  return false;
}

#ifdef MINIFE_NO_PINNED
// Fallback to ordinary pageable storage for A/B comparison.
template <typename T>
using PinnedVector = std::vector<T>;
#else
template <typename T>
using PinnedVector = std::vector<T, PinnedAllocator<T>>;
#endif

} // namespace miniFE

#endif
