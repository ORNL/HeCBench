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
#include <omp.h>
#include "scan.h"
#include <cstdlib>

void bScan(const unsigned int blockSize,
           const unsigned int len,
           const float *input,
           float *output,
           float *sumBuffer)
{
  #pragma omp target teams num_teams(len/blockSize) thread_limit(blockSize/2)
  {
    float block[256];
    #pragma omp parallel 
    {
      int tid = omp_get_thread_num();
      int bid = omp_get_team_num();
      int gid = bid * blockSize/2 + tid;
      
      /* Cache the computational window in shared memory */
      block[2*tid]     = input[2*gid];
      block[2*tid + 1] = input[2*gid + 1];
      #pragma omp barrier
      
      float cache0 = block[0];
      float cache1 = cache0 + block[1];
      
      /* build the sum in place up the tree */
      for(int stride = 1; stride < blockSize; stride *=2) {
        if(2*tid>=stride) {
          cache0 = block[2*tid-stride]+block[2*tid];
          cache1 = block[2*tid+1-stride]+block[2*tid+1];
        }
        #pragma omp barrier
      
        block[2*tid] = cache0;
        block[2*tid+1] = cache1;
      
        #pragma omp barrier
      }
      
      /* store the value in sum buffer before making it to 0 */   
      sumBuffer[bid] = block[blockSize-1];
      
      /*write the results back to global memory */
      if(tid==0) {
        output[2*gid]     = 0;
        output[2*gid+1]   = block[2*tid];
      } else {
        output[2*gid]     = block[2*tid-1];
        output[2*gid + 1] = block[2*tid];
      }
    }
  }
}

void pScan(const unsigned int blockSize,
           const unsigned int len,
           const float *input,
           float *output)
{
  #pragma omp target teams num_teams(1) thread_limit(len/2)
  {
    // float block[len+1];
    float block[4];
    #pragma omp parallel 
    {
      int tid = omp_get_thread_num();
      int bid = omp_get_team_num();
      int gid = bid * len/2 + tid;

      /* Cache the computational window in shared memory */
      block[2*tid]     = input[2*gid];
      block[2*tid + 1] = input[2*gid + 1];
      #pragma omp barrier

      float cache0 = block[0];
      float cache1 = cache0 + block[1];

      /* build the sum in place up the tree */
      for(int stride = 1; stride < blockSize; stride *=2) {

        if(2*tid>=stride) {
          cache0 = block[2*tid-stride]+block[2*tid];
          cache1 = block[2*tid+1-stride]+block[2*tid+1];
        }
        #pragma omp barrier

        block[2*tid] = cache0;
        block[2*tid+1] = cache1;

        #pragma omp barrier
      }

      /*write the results back to global memory */
      if(tid==0) {
        output[2*gid]     = 0;
        output[2*gid+1]   = block[2*tid];
      } else {
        output[2*gid]     = block[2*tid-1];
        output[2*gid + 1] = block[2*tid];
      }
    }
  }
}

void bAddition(const unsigned int blockSize,
               const unsigned int len,
               const float *input,
               float *output)
{
  #pragma omp target teams num_teams(len/blockSize) thread_limit(blockSize)
  {
    float value;
    #pragma omp parallel 
    {
      int tid = omp_get_thread_num();
      int bid = omp_get_team_num();
      int gid = bid * blockSize + tid;
      /* Only 1 thread of a group will read from global buffer */
      if(tid == 0) value = input[bid];
      #pragma omp barrier

      output[gid] += value;
    }
  }
}


/*
 * Scan for verification
 */
void scanLargeArraysCPUReference(
    float * output,
    float * input,
    const unsigned int length)
{
  output[0] = 0;

  for(unsigned int i = 1; i < length; ++i)
  {
    output[i] = input[i-1] + output[i-1];
  }
}


int main(int argc, char * argv[])
{
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <repeat> <input length> <block size>\n";
    return 1;
  }
  int iterations = atoi(argv[1]);
  int length = atoi(argv[2]);
  int blockSize = atoi(argv[3]);

  if(iterations < 1)
  {
    std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }
  if(!isPowerOf2(length))
  {
    length = roundToPowerOf2(length);
  }

  if((length/blockSize>GROUP_SIZE)&&(((length)&(length-1))!=0))
  {
    std::cout << "Invalid length: " << length << std::endl;
    return -1;
  }

  // input buffer size
  unsigned int sizeBytes = length * sizeof(float);

  float* inputBuffer = (float*) malloc (sizeBytes);

  // random initialisation of input
  fillRandom<float>(inputBuffer, length, 1, 0, 255);

  blockSize = (blockSize < length/2) ? blockSize : length/2;

  // Calculate number of passes required
  float t = std::log((float)length) / std::log((float)blockSize);
  unsigned int pass = (unsigned int)t;

  // If t is equal to pass
  if(std::fabs(t - (float)pass) < 1e-7)
  {
    pass--;
  }
  
  // Allocate 1D output buffer
  int outputBufferSize = 0;
  int* outputBufferSizeOffset = (int*) malloc (sizeof(int) * pass);
  for(unsigned int i = 0; i < pass; i++)
  {
    outputBufferSizeOffset[i] = outputBufferSize;
    outputBufferSize += (int)(length / std::pow((float)blockSize,(float)i));
  }

  float* outputBuffer = (float*) malloc (sizeof(float) * outputBufferSize);

  // Allocate 1D blockSumBuffer
  int blockSumBufferSize = 0;
  int* blockSumBufferSizeOffset = (int*) malloc (sizeof(int) * pass);
  for(unsigned int i = 0; i < pass; i++)
  {
    blockSumBufferSizeOffset[i] = blockSumBufferSize;
    blockSumBufferSize += (int)(length / std::pow((float)blockSize,(float)(i + 1)));
  }
  float* blockSumBuffer = (float*) malloc (sizeof(float) * blockSumBufferSize);

  // Allocate a tempBuffer 
  int tempLength = (int)(length / std::pow((float)blockSize, (float)pass));
  float* tempBuffer = (float*) malloc (sizeof(float) * tempLength);

  std::cout << "Executing kernel for " << iterations << " iterations\n";
  std::cout << "-------------------------------------------\n";

#pragma omp target data map(to: inputBuffer[0:length]) \
                        map(alloc: tempBuffer[0:tempLength], \
                                   blockSumBuffer[0:blockSumBufferSize], \
                                   outputBuffer[0:outputBufferSize])
{
  auto start = std::chrono::steady_clock::now();

  for(int n = 0; n < iterations; n++)
  {
    // Do block-wise sum
    bScan(blockSize, length, inputBuffer, 
          outputBuffer + outputBufferSizeOffset[0], 
          blockSumBuffer + blockSumBufferSizeOffset[0]);

    for(int i = 1; i < (int)pass; i++)
    {
      int size = (int)(length / std::pow((float)blockSize,(float)i));
      bScan(blockSize, size, blockSumBuffer + blockSumBufferSizeOffset[i - 1], 
            outputBuffer + outputBufferSizeOffset[i], 
            blockSumBuffer + blockSumBufferSizeOffset[i]);
    }

    // Do scan to tempBuffer
    pScan(blockSize, tempLength, 
          blockSumBuffer + blockSumBufferSizeOffset[pass - 1], tempBuffer);

    // Do block-addition on outputBuffers
    bAddition(blockSize, (unsigned int)(length / std::pow((float)blockSize, (float)(pass - 1))),
          tempBuffer, 
          outputBuffer + outputBufferSizeOffset[pass - 1]);

    for(int i = pass - 1; i > 0; i--)
    {
      bAddition(blockSize, (unsigned int)(length / std::pow((float)blockSize, (float)(i - 1))),
            outputBuffer + outputBufferSizeOffset[i], 
            outputBuffer + outputBufferSizeOffset[i - 1]);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of scan kernels: " << time * 1e-3f / iterations
            << " (us)\n";

  #pragma omp target update from (outputBuffer[0: \
     (pass==1) ? outputBufferSize : outputBufferSizeOffset[1]])
}

  // verification
  float* verificationOutput = (float*)malloc(sizeBytes);
  memset(verificationOutput, 0, sizeBytes);

  // reference implementation
  scanLargeArraysCPUReference(verificationOutput, inputBuffer, length);

  // compare the results and see if they match
  if (compare<float>(outputBuffer, verificationOutput, length, (float)0.001))
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;

  free(verificationOutput);
  free(inputBuffer);
  free(tempBuffer);
  free(blockSumBuffer);
  free(blockSumBufferSizeOffset);
  free(outputBuffer);
  free(outputBufferSizeOffset);
  if (!(compare<float>(outputBuffer, verificationOutput, length, (float)0.001))) return 1;
  return 0;
}
