#ifndef MTTKRP_GPU_H
#define MTTKRP_GPU_H

#include "util.h"
#if defined(__HIPCC__)
  #include <hip/hip_runtime.h>
#else
  #include <cuda.h>
#endif

int MTTKRP_MIHCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt);

int init_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt, ITYPE **dInds2, ITYPE **dfbrPtr1, ITYPE **dfbrIdx1, ITYPE **dFbrLikeSlcInds, DTYPE **dVals, DTYPE **dU);


#endif
