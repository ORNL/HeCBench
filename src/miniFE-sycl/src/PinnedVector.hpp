#ifndef _PinnedVector_hpp_
#define _PinnedVector_hpp_

// A std::allocator drop-in that backs storage with SYCL page-locked (pinned)
// host USM (sycl::malloc_host). Data placed here is transferred to the device
// at full PCIe bandwidth instead of the reduced rate the driver achieves when
// it has to stage pageable memory through an internal pinned buffer. Used for
// the large matrix (row_offsets/packed_cols/packed_coefs) and vector
// coefficient arrays copied host->device once before the CG solve.
//
// malloc_host requires a queue/context, and the host allocation must be used by
// a queue sharing that context. We therefore expose a single process-wide
// in-order queue (minife_pinned_queue) that both this allocator and cg_solve
// use, so the setup memcpy sees these pointers as genuine host USM.

#include <cstddef>
#include <new>
#include <vector>
#include <sycl/sycl.hpp>

namespace miniFE {

inline sycl::queue &minife_pinned_queue() {
  static sycl::queue q(
#ifdef USE_GPU
      sycl::gpu_selector_v,
#else
      sycl::cpu_selector_v,
#endif
      sycl::property::queue::in_order());
  return q;
}

template <typename T>
struct PinnedAllocator {
  using value_type = T;

  PinnedAllocator() noexcept = default;
  template <typename U>
  PinnedAllocator(const PinnedAllocator<U> &) noexcept {}

  T *allocate(std::size_t n) {
    if (n == 0)
      return nullptr;
    T *p = sycl::malloc_host<T>(n, minife_pinned_queue());
    if (p == nullptr)
      throw std::bad_alloc();
    return p;
  }

  void deallocate(T *p, std::size_t) noexcept {
    if (p)
      sycl::free(p, minife_pinned_queue());
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
