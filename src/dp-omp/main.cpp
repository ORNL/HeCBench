/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// *********************************************************************
// A simple demo application that implements a
// vector dot product computation between 2 T arrays.
//
// Runs computations with on the GPU device and then checks results
// against basic host CPU/C++ computation.
// *********************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <random>
#include "shrUtils.h"

template <typename T>
void dot (const size_t iNumElements, const int iNumIterations)
{
  // set and log Global and Local work size dimensions
  int szLocalWorkSize = 256;
  size_t szGlobalWorkSize = shrRoundUp(szLocalWorkSize, iNumElements);

  printf("Global Work Size \t\t= %zu\nLocal Work Size \t\t= %d\n",
         szGlobalWorkSize, szLocalWorkSize);

  const size_t src_size = szGlobalWorkSize;
  const size_t src_size_bytes = src_size * sizeof(T);

  // Allocate and initialize host arrays
  T* srcA = (T*) malloc (src_size_bytes);
  T* srcB = (T*) malloc (src_size_bytes);
  T dst, dst_ref = 0;

  size_t i;
  std::mt19937 engine(19937);
  std::uniform_int_distribution<int> dis (-32, 32);

  for (i = 0; i < iNumElements; ++i)
  {
    srcA[i] = dis(engine);
    srcB[i] = dis(engine);
    dst_ref += srcA[i] * srcB[i];
  }
  for (i = iNumElements; i < src_size ; ++i) {
    srcA[i] = srcB[i] = 0;
  }

  #pragma omp target data map(to: srcA[0:src_size], srcB[0:src_size])
  {
    // warmup
    for (i = 0; i < 100; i++) {
      dst = 0;
      #pragma omp target teams distribute parallel for map(tofrom: dst) \
       reduction(+:dst) thread_limit(szLocalWorkSize)
      for (size_t iGID = 0; iGID < src_size / 4; iGID++) {
        size_t iInOffset = iGID * 4;
        dst += srcA[iInOffset    ] * srcB[iInOffset    ] +
               srcA[iInOffset + 1] * srcB[iInOffset + 1] +
               srcA[iInOffset + 2] * srcB[iInOffset + 2] +
               srcA[iInOffset + 3] * srcB[iInOffset + 3];
      }
    }
    auto start = std::chrono::steady_clock::now();

    for (i = 0; i < (size_t)iNumIterations; i++) {
      dst = 0;
      #pragma omp target teams distribute parallel for map(tofrom: dst) \
       reduction(+:dst) thread_limit(szLocalWorkSize)
      for (size_t iGID = 0; iGID < src_size / 4; iGID++) {
        size_t iInOffset = iGID * 4;
        dst += srcA[iInOffset    ] * srcB[iInOffset    ] +
               srcA[iInOffset + 1] * srcB[iInOffset + 1] +
               srcA[iInOffset + 2] * srcB[iInOffset + 2] +
               srcA[iInOffset + 3] * srcB[iInOffset + 3];
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);
    printf("%s\n\n", dst == dst_ref ? "PASS" : "FAIL");
  }

  // Compute and compare results for golden-host and report errors and pass/fail

  free(srcA);
  free(srcB);
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t iNumElements = atol(argv[1]);
  const int iNumIterations = atoi(argv[2]);

  printf("------------- Data type is Float32 ---------------\n");
  dot<float>(iNumElements, iNumIterations);
  printf("------------- Data type is Float64 ---------------\n");
  dot<double>(iNumElements, iNumIterations);

  if (dst != dst_ref) return 1;
  return EXIT_SUCCESS;
}
