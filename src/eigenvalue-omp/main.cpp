/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  •   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  •   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>

#include "reference.h"
#include "utils.cpp"
#include "kernels.cpp"

void runKernels(
    float *diagonalBuffer,
    uint *numEigenValuesIntervalBuffer,
    float *offDiagonalBuffer,
    float *eigenIntervalBuffer0,
    float *eigenIntervalBuffer1,

    // reset the eigenvalue intervals buffer
    float **eigenIntervals,

    const int length,
    const float tolerance,
    // index of the two eigenInterval buffers
    uint &in )
{
  #pragma omp target update to (eigenIntervalBuffer0[0:length*2])
  #pragma omp target update to (eigenIntervalBuffer1[0:length*2])

  in = 0;
  while (isComplete(eigenIntervals[in], length, tolerance)) {

    calNumEigenValueInterval(
      numEigenValuesIntervalBuffer,
      eigenIntervalBuffer0,
      diagonalBuffer,
      offDiagonalBuffer,
      length);

    recalculateEigenIntervals(
      eigenIntervalBuffer1,
      eigenIntervalBuffer0,
      numEigenValuesIntervalBuffer,
      diagonalBuffer,
      offDiagonalBuffer,
      length,
      tolerance);

    in = 1 - in;
    float *e = eigenIntervalBuffer1;
    eigenIntervalBuffer1 = eigenIntervalBuffer0,
    eigenIntervalBuffer0 = e;

    #pragma omp target update from (eigenIntervalBuffer0[0:length*2])
  }
}

int main(int argc, char * argv[])
{
  if (argc != 3) {
    printf("Usage: %s <length of the diagonal of the square matrix> <repeat>\n", argv[0]);
    return 1;
  }

  // Length of the diagonal of the square matrix
  int length = atoi(argv[1]);
  // Number of iterations for kernel execution
  int iterations = atoi(argv[2]);
  // Seed value for random number generation 
  uint seed = 123;
  float tolerance;
  // diagonal elements of the matrix
  float *diagonal;
  // off-diagonal elements of the matrix
  float *offDiagonal;
  // calculated eigen values of the matrix
  float *eigenIntervals[2];
  // index to one of the two eigen interval buffers
  uint  in;
  // eigen values using reference implementation
  float *verificationEigenIntervals[2];
  // index to one of the two eigen interval arrays
  uint   verificationIn;

  // allocate memory for diagonal elements of the matrix  of size lengthxlength

  if(isPowerOf2(length))
  {
    length = roundToPowerOf2(length);
  }

  if(length < 256)
  {
    length = 256;
  }

  uint diagonalSizeBytes = length * sizeof(float);
  diagonal = (float *) malloc(diagonalSizeBytes);
  CHECK_ALLOCATION(diagonal, "Failed to allocate host memory. (diagonal)");

  // allocate memory for offdiagonal elements of the matrix of length (length-1)
  uint offDiagonalSizeBytes = (length - 1) * sizeof(float);
  offDiagonal = (float *) malloc(offDiagonalSizeBytes);
  CHECK_ALLOCATION(offDiagonal, "Failed to allocate host memory. (offDiagonal)");

  /*
   * allocate memory to store the eigenvalue intervals interleaved with upperbound followed
   * by the lower bound interleaved
   * An array of two is used for using it for two different passes
   */
  uint eigenIntervalsSizeBytes = (2*length) * sizeof(float);
  for(int i = 0; i < 2; ++i)
  {
    eigenIntervals[i] = (float *) malloc(eigenIntervalsSizeBytes);
    CHECK_ALLOCATION(eigenIntervals[i],
        "Failed to allocate host memory. (eigenIntervals)");
  }

  // random initialisation of input using a seed
  fillRandom<float>(diagonal   , length  , 1, 0, 255, seed);
  fillRandom<float>(offDiagonal, length-1, 1, 0, 255, seed+10);

  // calculate the upperbound and the lowerbound of the eigenvalues of the matrix
  float lowerLimit;
  float upperLimit;
  computeGerschgorinInterval(&lowerLimit, &upperLimit, diagonal, offDiagonal, length);

  // initialize the eigenvalue intervals
  eigenIntervals[0][0]= lowerLimit;
  eigenIntervals[0][1]= upperLimit;

  // the following intervals have no eigenvalues
  for (int i = 2 ; i < 2*length ; i++)
  {
    eigenIntervals[0][i] = upperLimit;
  }

  tolerance = 0.001f;
  /*
   * Unless quiet mode has been enabled, print the INPUT array.
   */
#ifdef DEBUG
  printArray<float>("Diagonal", diagonal, length, 1);
  printArray<float>("offDiagonal", offDiagonal, length-1, 1);
#endif

  // store the diagonal elements of the matrix
  float *diagonalBuffer = diagonal;

  // store the number of eigenvalues in each interval
  uint *numEigenValuesIntervalBuffer = (uint*) malloc (sizeof(uint) * length);

  // store the offDiagonal elements of the matrix
  float *offDiagonalBuffer = offDiagonal;

  // store the eigenvalue intervals
  //float *eigenIntervalBuffer = (float*) malloc (sizeof(float) * length * 4);
  float *eigenIntervalBuffer0 = eigenIntervals[0];
  float *eigenIntervalBuffer1 = eigenIntervals[1];

#pragma omp target data map(to: diagonalBuffer[0:length], \
                                offDiagonalBuffer[0:length-1]) \
                        map(alloc: numEigenValuesIntervalBuffer[0:length], \
                                   eigenIntervalBuffer0[0:length*2], \
                                   eigenIntervalBuffer1[0:length*2])
{
  // Warm up
  for(int i = 0; i < 2 && iterations != 1; i++)
  {
    // Arguments are set and execution call is enqueued on command buffer
    runKernels(
        diagonalBuffer,
        numEigenValuesIntervalBuffer,
        offDiagonalBuffer,
        eigenIntervalBuffer0,
        eigenIntervalBuffer1,
        eigenIntervals,   // reset eigenIntervals
        length,
        tolerance,
        in
        );
  }

  std::cout << "Executing kernel for " << iterations
            << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
  {
    runKernels(
        diagonalBuffer,
        numEigenValuesIntervalBuffer,
        offDiagonalBuffer,
        eigenIntervalBuffer0,
        eigenIntervalBuffer1,
        eigenIntervals,   // reset eigenIntervals
        length,
        tolerance,
        in
        );
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-3f) / iterations << " (us)\n";
}

  // Verify results
  for(int i = 0 ; i < 2; ++i)
  {
    verificationEigenIntervals[i] = (float *) malloc(eigenIntervalsSizeBytes);

    if(verificationEigenIntervals[i] == NULL)
    {
      error("Failed to allocate host memory. (verificationEigenIntervals)");
      return 1;
    }
  }

  computeGerschgorinInterval(&lowerLimit, &upperLimit, diagonal, offDiagonal, length);

  verificationIn = 0;
  verificationEigenIntervals[verificationIn][0]= lowerLimit;
  verificationEigenIntervals[verificationIn][1]= upperLimit;

  for(int i = 2 ; i < 2*length ; i++)
  {
    verificationEigenIntervals[verificationIn][i] = upperLimit;
  }

  while(isComplete(verificationEigenIntervals[verificationIn], length, tolerance))
  {
    eigenValueCPUReference(diagonal,offDiagonal, length,
        verificationEigenIntervals[verificationIn],
        verificationEigenIntervals[1-verificationIn],
        tolerance);
    verificationIn = 1 - verificationIn;
  }

  // select the buffers for comparison
  if(compare(eigenIntervals[in], 
             verificationEigenIntervals[verificationIn], 2*length))
  {
    std::cout<<"PASS\n" << std::endl;
  }
  else
  {
    std::cout<<"FAIL\n" << std::endl;
    exit(1);
  }

  // release program resources
  free(diagonal);
  free(offDiagonal);
  free(eigenIntervals[0]);
  free(eigenIntervals[1]);
  free(verificationEigenIntervals[0]);
  free(verificationEigenIntervals[1]);
  free(numEigenValuesIntervalBuffer);

  return 0;
}
