#pragma once

#include <sycl/sycl.hpp>

// Fused fp32 path for one batch of 128x128 problems:
//   T = diag(alpha) * (I + A)^{-1}
//   J = B * (I + A)^{-1}
//
// A is symmetric PSD, stored row-major and fully populated. B and alpha are
// row-major too. One launch, one work-group per matrix. A and B are not
// modified. nmat is the nd_range size and is not a template.
//
// The sub-group size is the largest the device lists, when that size is 32 or
// 64. One lane owns one column of a tile of that size. 128 and the 256-item
// work-group divide both. Local memory follows local_layout.hpp, the same
// rule as CUDA and HIP. The host entry points match those ports, plus the
// queue: 0 on success, 1 on failure.
//
// fused_prepare queries the device once, the step CUDA and HIP do inside
// fused_layout and fused_init. It rejects a device whose sub-group or local
// memory does not fit this binary's kernel.
int fused_prepare(const sycl::device& dev);
int fused_init(sycl::queue& q, int nmat);
void fused_shutdown(sycl::queue& q);
int fused_launch(sycl::queue& q, const float* A, const float* B, const float* alpha,
                 float* T, float* J, int nmat);
// 1 if any Cholesky pivot was not strictly positive. The flag is cleared in
// fused_init and is read after the caller has synchronized.
int fused_pivot_failed(sycl::queue& q);
int fused_wave();
const char* fused_layout();
