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

///////////////////////////////////////////////////////////////////////////////
// This sample implements Niederreiter quasirandom number generator
// and Moro's Inverse Cumulative Normal Distribution generator
///////////////////////////////////////////////////////////////////////////////

// standard utilities and systems includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "qrg.h"

// forward declarations
void initQuasirandomGenerator(unsigned int *table);
double getQuasirandomValue63(INT64 i, int dim);
double MoroInvCNDcpu(unsigned int x);

////////////////////////////////////////////////////////////////////////////////
// Moro's Inverse Cumulative Normal Distribution function approximation
////////////////////////////////////////////////////////////////////////////////
#pragma omp declare target
float MoroInvCNDgpu(unsigned int x)
{
  const float a1 = 2.50662823884f;
  const float a2 = -18.61500062529f;
  const float a3 = 41.39119773534f;
  const float a4 = -25.44106049637f;
  const float b1 = -8.4735109309f;
  const float b2 = 23.08336743743f;
  const float b3 = -21.06224101826f;
  const float b4 = 3.13082909833f;
  const float c1 = 0.337475482272615f;
  const float c2 = 0.976169019091719f;
  const float c3 = 0.160797971491821f;
  const float c4 = 2.76438810333863E-02f;
  const float c5 = 3.8405729373609E-03f;
  const float c6 = 3.951896511919E-04f;
  const float c7 = 3.21767881768E-05f;
  const float c8 = 2.888167364E-07f;
  const float c9 = 3.960315187E-07f;

  float z;

  bool negate = false;

  // Ensure the conversion to floating point will give a value in the
  // range (0,0.5] by restricting the input to the bottom half of the
  // input domain. We will later reflect the result if the input was
  // originally in the top half of the input domain
  if (x >= 0x80000000UL)
  {
    x = 0xffffffffUL - x;
    negate = true;
  }

  // x is now in the range [0,0x80000000) (i.e. [0,0x7fffffff])
  // Convert to floating point in (0,0.5]
  const float x1 = 1.0f / (float)0xffffffffUL;
  const float x2 = x1 / 2.0f;
  float p1 = x * x1 + x2;
  // Convert to floating point in (-0.5,0]
  float p2 = p1 - 0.5f;

  // The input to the Moro inversion is p2 which is in the range
  // (-0.5,0]. This means that our output will be the negative side
  // of the bell curve (which we will reflect if "negate" is true).

  // Main body of the bell curve for |p| < 0.42
  if (p2 > -0.42f)
  {
    z = p2 * p2;
    z = p2 * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0f);
  }
  // Special case (Chebychev) for tail
  else
  {
    z = logf(-logf(p1));
    z = - (c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9))))))));
  }

  // If the original input (x) was in the top half of the range, reflect
  // to get the positive side of the bell curve
  return negate ? -z : z;
}
#pragma omp end declare target

// size of output random array
const unsigned int N = 1048576;

///////////////////////////////////////////////////////////////////////////////
// Wrapper for Niederreiter quasirandom number generator kernel
///////////////////////////////////////////////////////////////////////////////
void QuasirandomGeneratorGPU(float* output,
                             const unsigned int* table,
                             const unsigned int seed,
                             const unsigned int N,
                             const size_t szWorkgroup)
{
}

///////////////////////////////////////////////////////////////////////////////
// Wrapper for Inverse Cumulative Normal Distribution generator kernel
///////////////////////////////////////////////////////////////////////////////
void InverseCNDGPU(float* output, 
                   const unsigned int pathN,
                   const size_t szWorkgroup)
{
}

int main(int argc, const char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned int dim, pos;
  double delta, ref, sumDelta, sumRef, L1norm;
  unsigned int table[QRNG_DIMENSIONS*QRNG_RESOLUTION];
  bool bPassFlag = false;

  float* output = (float *)malloc(QRNG_DIMENSIONS * N * sizeof(float));

  printf("Initializing QRNG tables...\n");
  initQuasirandomGenerator(table);

  printf(">>>Launch QuasirandomGenerator kernel...\n\n"); 

  size_t szWorkgroup = 64 * (256 / QRNG_DIMENSIONS)/64;

  #pragma omp target data map(alloc: output[0:QRNG_DIMENSIONS*N]) \
                          map(to: table[0:QRNG_DIMENSIONS*QRNG_RESOLUTION])
  {
    // seed is fixed at zero
    const unsigned int seed = 0;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
    {
      #pragma omp target teams distribute parallel for collapse(2) thread_limit(szWorkgroup)
      for (unsigned int pos = 0; pos < N; pos++) {
        for (unsigned int y = 0; y < QRNG_DIMENSIONS; y++) {
          unsigned int result = 0;
          unsigned int data = seed + pos;
          for(int bit = 0; bit < QRNG_RESOLUTION; bit++, data >>= 1)
            if(data & 1) result ^= table[bit + y * QRNG_RESOLUTION];
          output[y * N + pos] = (float)(result + 1) * INT_SCALE;
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (qrng): %f (us)\n", (time * 1e-3f) / repeat);

    printf("\nRead back results...\n"); 
    #pragma omp target update from (output[0: QRNG_DIMENSIONS*N])

    printf("Comparing to the CPU results...\n\n");
    sumDelta = 0;
    sumRef   = 0;
    for(dim = 0; dim < QRNG_DIMENSIONS; dim++)
    {
      for(pos = 0; pos < N; pos++)
      {
        ref       = getQuasirandomValue63(pos, dim);
        delta     = (double)output[dim * N  + pos] - ref;
        sumDelta += fabs(delta);
        sumRef   += fabs(ref);
      }
    }
    L1norm = sumDelta / sumRef;
    printf("  L1 norm: %E\n", L1norm);
    printf("  ckQuasirandomGenerator deviations %s Allowable Tolerance\n\n\n", (L1norm < 1e-6) ? "WITHIN" : "ABOVE");
    bPassFlag = (L1norm < 1e-6);

    printf(">>>Launch InverseCND kernel...\n\n"); 

    // determine work group sizes for each device
    szWorkgroup = 128;
    const unsigned int pathN = QRNG_DIMENSIONS * N;
    const unsigned int distance = ((unsigned int)-1) / (pathN  + 1);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
    {
      #pragma omp target teams distribute parallel for thread_limit(szWorkgroup)
      for(unsigned int pos = 0; pos < pathN; pos++){
        unsigned int d = (pos + 1) * distance;
        output[pos] = MoroInvCNDgpu(d);
      }
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (icnd): %f (us)\n", (time * 1e-3f) / repeat);
    printf("\nRead back results...\n"); 

    #pragma omp target update from (output[0: QRNG_DIMENSIONS*N])

    printf("Comparing to the CPU results...\n\n");

    sumDelta = 0;
    sumRef   = 0;
    for(pos = 0; pos < QRNG_DIMENSIONS * N; pos++){
      unsigned int d = (pos + 1) * distance;
      ref       = MoroInvCNDcpu(d);
      delta     = (double)output[pos] - ref;
      sumDelta += fabs(delta);
      sumRef   += fabs(ref);
    }
    L1norm = sumDelta / sumRef;
    printf("  L1 norm: %E\n", L1norm);
    printf("  ckInverseCNDGPU deviations %s Allowable Tolerance\n\n\n", (L1norm < 1e-6) ? "WITHIN" : "ABOVE");
    bPassFlag &= (L1norm < 1e-6);

    if (bPassFlag)
      printf("PASS\n");
    else
      printf("FAIL\n");
    if (!bPassFlag) exit(1);

    free(output);
  }
  return 0;
}
