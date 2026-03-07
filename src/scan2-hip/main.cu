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
#include <hip/hip_runtime.h>
#include "scan.h"
#include <cstdlib>

/*
 * ScanLargeArrays : Scan is done for each block and the sum of each
 * block is stored in separate array (sumBuffer). SumBuffer is scanned
 * and results are added to every value of next corresponding block to
 * compute the scan of a large array.(not limited to 2*MAX_GROUP_SIZE)
 * Scan uses a balanced tree algorithm. See Belloch, 1990 "Prefix Sums
 * and Their Applications"
 * @param output output data 
 * @param input  input data
 * @param block  local memory used in the kernel
 * @param sumBuffer  sum of blocks
 * @param length length of the input data
 */

__global__
void blockAddition(const float*__restrict__ input,
                         float*__restrict__ output)
{  
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int gid = bid * blockDim.x + tid;

  __shared__ float value;

  /* Only 1 thread of a group will read from global buffer */
  if(tid == 0)
  {
    value = input[bid];
  }
  __syncthreads();

  output[gid] += value;
}

__global__
void ScanLargeArrays(float *__restrict__ output,
                     const float *__restrict__ input,
                     const unsigned int block_size,   // size of block
                     float *__restrict__ sumBuffer)  // sum of blocks
{
  extern __shared__ float block[];   // Size : block_size
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;

  /* Cache the computational window in shared memory */
  block[2*tid]     = input[2*gid];
  block[2*tid + 1] = input[2*gid + 1];
  __syncthreads();

  float cache0 = block[0];
  float cache1 = cache0 + block[1];

  /* build the sum in place up the tree */
  for(int stride = 1; stride < block_size; stride *=2)
  {
    if(2*tid>=stride)
    {
      cache0 = block[2*tid-stride]+block[2*tid];
      cache1 = block[2*tid+1-stride]+block[2*tid+1];
    }
    __syncthreads();

    block[2*tid] = cache0;
    block[2*tid+1] = cache1;

    __syncthreads();
  }

  /* store the value in sum buffer before making it to 0 */   
  sumBuffer[bid] = block[block_size-1];

  /*write the results back to global memory */
  if(tid==0)
  {
    output[2*gid]     = 0;
    output[2*gid+1]   = block[2*tid];
  }
  else
  {
    output[2*gid]     = block[2*tid-1];
    output[2*gid + 1] = block[2*tid];
  }
}

__global__ 
void prefixSum(float *__restrict__ output, 
               const float *__restrict__ input,
               const unsigned int block_size)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;

  extern __shared__ float block[];

  /* Cache the computational window in shared memory */
  block[2*tid]     = input[2*gid];
  block[2*tid + 1] = input[2*gid + 1];
  __syncthreads();

  float cache0 = block[0];
  float cache1 = cache0 + block[1];

  /* build the sum in place up the tree */
  for(int stride = 1; stride < block_size; stride *=2)
  {

    if(2*tid>=stride)
    {
      cache0 = block[2*tid-stride]+block[2*tid];
      cache1 = block[2*tid+1-stride]+block[2*tid+1];
    }
    __syncthreads();

    block[2*tid] = cache0;
    block[2*tid+1] = cache1;

    __syncthreads();
  }

  /*write the results back to global memory */
  if(tid==0)
  {
    output[2*gid]     = 0;
    output[2*gid+1]   = block[2*tid];
  }
  else
  {
    output[2*gid]     = block[2*tid-1];
    output[2*gid + 1] = block[2*tid];
  }
}

void bScan(const unsigned int blockSize,
           const unsigned int len,
           const float *inputBuffer,
           float *outputBuffer,
           float *blockSumBuffer)
{
  // set the block size
  dim3 grid (len / blockSize);
  dim3 block (blockSize / 2);

  ScanLargeArrays<<<grid, block, sizeof(float)*blockSize>>>(
      outputBuffer, inputBuffer, blockSize, blockSumBuffer);
}

void pScan(const unsigned int blockSize,
           const unsigned int len,
           const float *inputBuffer,
           float *outputBuffer)
{
  dim3 grid (1);
  dim3 block (len / 2);
  hipLaunchKernelGGL(prefixSum, grid, block, (len+1)*sizeof(float), 0, outputBuffer, inputBuffer, blockSize);
}

void bAddition(const unsigned int blockSize,
    const unsigned int len,
    float *inputBuffer,
    float *outputBuffer)
{
  // set the block size
  dim3 grid (len / blockSize);
  dim3 block (blockSize);
  hipLaunchKernelGGL(blockAddition, grid, block, 0, 0, inputBuffer, outputBuffer);
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

  float* input = (float*) malloc (sizeBytes);

  // store device results for verification
  float* output = (float*) malloc (sizeBytes);

  // random initialisation of input
  fillRandom<float>(input, length, 1, 0, 255);

  blockSize = (blockSize < length/2) ? blockSize : length/2;

  // Calculate number of passes required
  float t = std::log((float)length) / std::log((float)blockSize);
  unsigned int pass = (unsigned int)t;

  // If t is equal to pass
  if(std::fabs(t - (float)pass) < 1e-7)
  {
    pass--;
  }

  // Create input buffer on device
  float* inputBuffer; 
  hipMalloc((void**)&inputBuffer, sizeBytes);
  hipMemcpy(inputBuffer, input, sizeBytes, hipMemcpyHostToDevice);

  // Allocate output buffers
  std::vector<float*> outputBuffers(pass);

  for(unsigned int i = 0; i < pass; i++)
  {
    int size = (int)(length / std::pow((float)blockSize,(float)i));
    float* outputBuffer; 
    hipMalloc((void**)&outputBuffer, size * sizeof(float));
    outputBuffers[i] = outputBuffer;
  }

  // Allocate blockSumBuffers
  std::vector<float*> blockSumBuffers(pass);

  for(unsigned int i = 0; i < pass; i++)
  {
    int size = (int)(length / std::pow((float)blockSize,(float)(i + 1)));
    float* sum; 
    hipMalloc((void**)&sum, size * sizeof(float));
    blockSumBuffers[i] = sum;
  }

  // Create a tempBuffer on device
  int tempLength = (int)(length / std::pow((float)blockSize, (float)pass));

  float* tempBuffer; 
  hipMalloc((void**)&tempBuffer, tempLength * sizeof(float));

  std::cout << "Executing kernel for " << iterations << " iterations\n";
  std::cout << "-------------------------------------------\n";

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for(int n = 0; n < iterations; n++)
  {
    // Do block-wise sum
    bScan(blockSize, length, inputBuffer, outputBuffers[0], blockSumBuffers[0]);

    for(int i = 1; i < (int)pass; i++)
    {
      int size = (int)(length / std::pow((float)blockSize,(float)i));
      bScan(blockSize, size, blockSumBuffers[i - 1], outputBuffers[i], blockSumBuffers[i]);
    }

    // Do scan to tempBuffer
    pScan(blockSize, tempLength, blockSumBuffers[pass - 1], tempBuffer);

    // Do block-addition on outputBufferss
    bAddition(blockSize, (unsigned int)(length / std::pow((float)blockSize, (float)(pass - 1))),
        tempBuffer, outputBuffers[pass - 1]);

    for(int i = pass - 1; i > 0; i--)
    {
      bAddition(blockSize, (unsigned int)(length / std::pow((float)blockSize, (float)(i - 1))),
          outputBuffers[i], outputBuffers[i - 1]);
    }
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of scan kernels: " << time * 1e-3f / iterations
            << " (us)\n";

  hipMemcpy(output, outputBuffers[0], sizeBytes, hipMemcpyDeviceToHost);

  hipFree(inputBuffer); 

  for(unsigned int i = 0; i < pass; i++)
  {
    hipFree(outputBuffers[i]);
    hipFree(blockSumBuffers[i]);
  }

  hipFree(tempBuffer);

  // verification
  float* verificationOutput = (float*)malloc(sizeBytes);
  memset(verificationOutput, 0, sizeBytes);

  // reference implementation
  scanLargeArraysCPUReference(verificationOutput, input, length);

  // compare the results and see if they match
  if (compare<float>(output, verificationOutput, length, (float)0.001))
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;

  free(input);
  free(output);
  free(verificationOutput);
  if (!(compare<float>(output, verificationOutput, length, (float)0.001))) return 1;
  return 0;
}
