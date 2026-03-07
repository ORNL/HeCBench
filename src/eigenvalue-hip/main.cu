/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
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
#include <hip/hip_runtime.h>

#include "reference.h"
#include "utils.cu"
#include "kernels.cu"

void runKernels(
    const float *diagonalBuffer,
           uint *numEigenValuesIntervalBuffer,
    const float *offDiagonalBuffer,
    float **eigenIntervalBuffer,

    // reset the eigenvalue intervals buffer
    float **eigenIntervals,

    const int length,
    const float tolerance,
    // index of the two eigenInterval buffers
    uint &in )
{
  dim3 grid (length/256);
  dim3 block (256);

  for (int i = 0; i < 2; i++)
    hipMemcpy(eigenIntervalBuffer[i], eigenIntervals[i], length*2*sizeof(float),
               hipMemcpyHostToDevice);

  in = 0;
  while (isComplete(eigenIntervals[in], length, tolerance)) {

    hipLaunchKernelGGL(calNumEigenValueInterval, grid, block, 0, 0, 
            numEigenValuesIntervalBuffer,
            eigenIntervalBuffer[in],
            diagonalBuffer,
            offDiagonalBuffer,
            length);

    hipLaunchKernelGGL(recalculateEigenIntervals, grid, block, 0, 0, 
          eigenIntervalBuffer[1-in],
          eigenIntervalBuffer[in],
          numEigenValuesIntervalBuffer,
          diagonalBuffer,
          offDiagonalBuffer,
          length,
          tolerance);

    in = 1 - in;

    hipMemcpy(eigenIntervals[in], eigenIntervalBuffer[in], length*2*sizeof(float),
               hipMemcpyDeviceToHost);
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
  float *diagonalBuffer;
  hipMalloc((void**)&diagonalBuffer, sizeof(float) * length);
  hipMemcpy(diagonalBuffer, diagonal, sizeof(float) * length, hipMemcpyHostToDevice); 

  // store the number of eigenvalues in each interval
  uint *numEigenValuesIntervalBuffer;
  hipMalloc((void**)&numEigenValuesIntervalBuffer, sizeof(uint) * length);

  // store the offDiagonal elements of the matrix
  float *offDiagonalBuffer;
  hipMalloc((void**)&offDiagonalBuffer, sizeof(float) * (length-1));
  hipMemcpy(offDiagonalBuffer, offDiagonal, sizeof(float) * (length-1), hipMemcpyHostToDevice); 

  // store the eigenvalue intervals
  float *eigenIntervalBuffer[2];
  for(int i = 0 ; i < 2 ; ++ i)
  {
    hipMalloc((void**)&(eigenIntervalBuffer[i]), sizeof(float) * (length * 2));
  }

  // Warm up
  for(int i = 0; i < 2 && iterations != 1; i++)
  {
    // Arguments are set and execution call is enqueued on command buffer
    runKernels(
        diagonalBuffer,
        numEigenValuesIntervalBuffer,
        offDiagonalBuffer,
        eigenIntervalBuffer,
        eigenIntervals,   // reset eigenIntervals
        length,
        tolerance,
        in);
  }

  std::cout << "Executing kernel for " << iterations
            << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;
  
  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  
  for(int i = 0; i < iterations; i++)
  {
    runKernels(
        diagonalBuffer,
        numEigenValuesIntervalBuffer,
        offDiagonalBuffer,
        eigenIntervalBuffer,
        eigenIntervals,   // reset eigenIntervals
        length,
        tolerance,
        in);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-3f) / iterations << " (us)\n";

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
  hipFree(diagonalBuffer);
  hipFree(offDiagonalBuffer);
  hipFree(numEigenValuesIntervalBuffer);
  hipFree(eigenIntervalBuffer[0]);
  hipFree(eigenIntervalBuffer[1]);
  free(diagonal);
  free(offDiagonal);
  free(eigenIntervals[0]);
  free(eigenIntervals[1]);
  free(verificationEigenIntervals[0]);
  free(verificationEigenIntervals[1]);

  return 0;
}
