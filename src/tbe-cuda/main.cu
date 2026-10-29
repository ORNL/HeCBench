// Derived conceptually from FBGEMM's BSD-3-Clause licensed public table
// batched embedding implementation: https://github.com/pytorch/FBGEMM
#include "tbe_gpu_main.h"

int main(int argc, char** argv) {
  return tbe_gpu_main(argc, argv, "CUDA");
}
