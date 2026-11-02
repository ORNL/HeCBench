#pragma once

#include <hip/hip_runtime.h>

// Fused fp32 path for one batch of 128x128 problems:
//   T = diag(alpha) * (I + A)^{-1}
//   J = B * (I + A)^{-1}
//
// A is symmetric PSD, stored row-major and fully populated. B and alpha are
// row-major too. One launch, one block per matrix. A and B are not modified.
// nmat is the grid size and is not a template.
//
// The tile inverted by one wave equals the device wave size (32 on NVIDIA,
// 64 on CDNA). 128 and the 256-thread block divide both, so the same source
// is instantiated for either wave. Shared memory follows local_layout.hpp,
// the same rule as CUDA and SYCL. The host entry points match those ports:
// 0 on success, 1 on failure.
int fused_init(int nmat);
void fused_shutdown();
int fused_launch(const float* A, const float* B, const float* alpha,
                 float* T, float* J, int nmat);
// 1 if any Cholesky pivot was not strictly positive. The flag is cleared in
// fused_init and is read after the caller has synchronized.
int fused_pivot_failed();
int fused_wave();
const char* fused_layout();
