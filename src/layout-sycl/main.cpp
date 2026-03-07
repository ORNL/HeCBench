/**********************************************************************
* Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
*
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
*  other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* ********************************************************************/


#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define TREE_NUM 4096
#define TREE_SIZE 4096
#define GROUP_SIZE 256

struct AppleTree
{
  int apples[TREE_SIZE];
};

struct ApplesOnTrees
{
  int trees[TREE_NUM];
};

void AoSKernel(const AppleTree *__restrict trees,
               int *__restrict outBuf,
               int treeSize, sycl::nd_item<1> &item)
{
  uint gid = item.get_global_id(0);
  uint res = 0;
  for(int i = 0; i < treeSize; i++)
  {
    res += trees[gid].apples[i];
  }
  outBuf[gid] = res;
}


void SoAKernel(const ApplesOnTrees *__restrict applesOnTrees,
               int *__restrict outBuf,
               int treeSize, sycl::nd_item<1> &item)
{
  uint gid = item.get_global_id(0);
  uint res = 0;
  for(int i = 0; i < treeSize; i++)
  {
    res += applesOnTrees[i].trees[gid];
  }
  outBuf[gid] = res;
}

int main(int argc, char * argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int iterations = atoi(argv[1]); // Number of iterations for kernel execution
  const int treeSize = TREE_SIZE;
  const int treeNumber = TREE_NUM;
  bool fail = false;

  if(iterations < 1)
  {
    std::cout<<"Iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }

  if(treeNumber < GROUP_SIZE)
  {
    std::cout<<"treeNumber should be larger than the work group size"<<std::endl;
    return -1;
  }
  if(treeNumber % 256 !=0)
  {
    std::cout<<"treeNumber should be a multiple of 256"<<std::endl;
    return -1;
  }

  const int elements = treeSize * treeNumber;
  size_t inputSize = elements * sizeof(int);
  size_t outputSize = treeNumber * sizeof(int);

  // input data
  int* data = (int*) malloc (inputSize);

  // device result
  int *deviceResult = (int *)malloc(outputSize);

  // compute reference for verification
  int *reference = (int *)malloc(outputSize);
  memset(reference,0,outputSize);
  for(int i = 0; i < treeNumber; i++)
    for(int j = 0; j < treeSize; j++)
      reference[i] += i * treeSize + j;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  sycl::range<1> gws(treeNumber);
  sycl::range<1> lws(GROUP_SIZE);

  int *inputBuffer = sycl::malloc_device<int>(elements, q);
  int *outputBuffer = sycl::malloc_device<int>(treeNumber, q);

  //initialize aos data
  for (int i = 0; i < treeNumber; i++)
    for(int j = 0; j < treeSize; j++)
      data[j + i* treeSize] = j + i* treeSize;

  q.memcpy(inputBuffer, data, inputSize).wait();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class AoS>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        AoSKernel((AppleTree*)inputBuffer, outputBuffer, treeSize, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (AoS): "
            << (time * 1e-3f) / iterations << " (us)\n";

  q.memcpy(deviceResult, outputBuffer, outputSize).wait();

  for(int i = 0; i< treeNumber; i++)
  {
    if(deviceResult[i] != reference[i])
    {
      fail = true;
      break;
    }
  }

  if (fail)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";
  if (fail) exit(1);

  //initialize soa data
  for (int i = 0; i < treeNumber; i++)
    for(int j = 0; j < treeSize; j++)
      data[i + j* treeNumber] = j + i* treeSize;

  q.memcpy(inputBuffer, data, inputSize).wait();

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class SoA>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        SoAKernel((ApplesOnTrees*)inputBuffer, outputBuffer, treeSize, item);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (SoA): "
            << (time * 1e-3f) / iterations << " (us)\n";

  q.memcpy(deviceResult, outputBuffer, outputSize).wait();

  for(int i = 0; i< treeNumber; i++)
  {
    if(deviceResult[i] != reference[i])
    {
      fail = true;
      break;
    }
  }

  if (fail)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";
  if (fail) exit(1);

  sycl::free(inputBuffer, q);
  sycl::free(outputBuffer, q);
  free(deviceResult);
  free(reference);
  free(data);
  return 0;
}
