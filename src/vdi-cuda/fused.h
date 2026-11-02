#pragma once

#include <cuda_runtime.h>

// Fused fp32 path for one batch of 128x128 problems:
//   T = diag(alpha) * (I + A)^{-1}
//   J = B * (I + A)^{-1}
//
// A is symmetric PSD, stored row-major and fully populated. B and alpha are
// row-major too. One launch, one block per matrix. A and B are not modified.
// nmat is the grid size and is not a template.
//
// This kernel is the wave-32 NVIDIA form. Shared memory follows
// local_layout.hpp, the same rule as HIP and SYCL: 82432 bytes keeps ld 129
// and the 64x64 scratch on chip; 64KB keeps ld 128 and that scratch in global
// memory; below 64KB is rejected. The host entry points match those ports:
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
