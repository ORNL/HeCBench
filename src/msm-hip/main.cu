// HIP build of the MSM benchmark. hipcc compiles the CUDA launch syntax;
// the runtime calls switch on __HIPCC__ inside the shared source.
#include "../msm-cuda/main.cu"
