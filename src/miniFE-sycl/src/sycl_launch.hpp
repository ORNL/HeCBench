#ifndef _sycl_launch_hpp_
#define _sycl_launch_hpp_

#include <sycl/sycl.hpp>

// Kernel launch helpers.
//
// miniFE's CG solver submits many very small kernels (waxpby, daxpby, dot,
// matvec) per iteration on an in-order queue. With the default q.submit() path
// each launch creates, records and destroys a device event, even though the
// results are only consumed via q.wait()/memcpy rather than per-kernel events.
// When USE_ENQUEUE_FUNCTIONS is defined we instead use the event-less
// sycl_ext_oneapi_enqueue_functions extension, which skips that per-launch
// event bookkeeping and closes much of the small-kernel overhead gap seen
// against the HIP port on MI300A.

namespace miniFE {

// Launch a basic nd-range kernel (no local memory).
template <typename KernelFunc>
static inline void launch_nd(sycl::queue &q, sycl::nd_range<1> ndr, KernelFunc k) {
#ifdef USE_ENQUEUE_FUNCTIONS
  sycl::ext::oneapi::experimental::nd_launch(q, ndr, k);
#else
  q.submit([&](sycl::handler &h) { h.parallel_for(ndr, k); });
#endif
}

// Launch an nd-range kernel that needs a single local (shared) accessor of
// Scalar. The kernel functor is invoked as k(nd_item<1>, Scalar* local_ptr).
template <typename Scalar, typename KernelFunc>
static inline void launch_nd_local(sycl::queue &q, sycl::nd_range<1> ndr,
                                   size_t local_size, KernelFunc k) {
#ifdef USE_ENQUEUE_FUNCTIONS
  namespace exp = sycl::ext::oneapi::experimental;
  exp::submit(q, [&](sycl::handler &h) {
    sycl::local_accessor<Scalar, 1> red(sycl::range<1>(local_size), h);
    exp::nd_launch(h, ndr, [=](sycl::nd_item<1> item) {
      k(item, red.template get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
#else
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<Scalar, 1> red(sycl::range<1>(local_size), h);
    h.parallel_for(ndr, [=](sycl::nd_item<1> item) {
      k(item, red.template get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
#endif
}

}//namespace miniFE

#endif
