// HIP entry point. hipcc sets __HIP__, so the included CUDA source takes the
// HIP runtime, bf16 type, and shuffle intrinsics. The kernels and the driver
// are otherwise the same translation unit.
#include "../rmsnorm-rope-cuda/main.cu"
