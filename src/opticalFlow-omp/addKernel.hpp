/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 */
#include "common.h"

static void Add(const float *op1, const float *op2, int count, float *sum) {
  #pragma omp target teams distribute parallel for
  for (int pos = 0; pos < count; ++pos) {
    sum[pos] = op1[pos] + op2[pos];
  }
}
