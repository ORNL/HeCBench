/**********************************************************************
  Copyright 2013 Advanced Micro Devices, Inc. All rights reserved.

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

#include "hwt.h"
#include <cstdlib>

// rounds to a power of 2
template<typename T>
T roundToPowerOf2(T val)
{
  int bytes = sizeof(T);
  val--;
  for(int i = 0; i < bytes; i++)
    val |= val >> (1<<i);
  val++;
  return val;
}

int main(int argc, char * argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <signal length> <repeat>\n";
    return 1;
  }
  unsigned int signalLength = atoi(argv[1]);
  const int iterations = atoi(argv[2]);

  // signal length must be power of 2
  signalLength = roundToPowerOf2<unsigned int>(signalLength);

  unsigned int levels = 0;
  if (getLevels(signalLength, &levels) == 1) {
    std::cerr << "signalLength > 2 ^ 23 not supported\n";
    return 1;
  }

  // Allocate and init memory used by host
  float *inData = (float*)malloc(signalLength * sizeof(float));

  srand(2);
  for(unsigned int i = 0; i < signalLength; i++)
  {
    inData[i] = (float)(rand() % 10);
  }

  float *dOutData = (float*) malloc(signalLength * sizeof(float));

  memset(dOutData, 0, signalLength * sizeof(float));

  float *dPartialOutData = (float*) malloc(signalLength * sizeof(float));

  memset(dPartialOutData, 0, signalLength * sizeof(float));

  float *hOutData = (float*)malloc(signalLength * sizeof(float));

  memset(hOutData, 0, signalLength * sizeof(float));

  float *inDataBuf;
  hipMalloc((void**)&inDataBuf, sizeof(float) * signalLength);

  float *partialOutDataBuf;
  hipMalloc((void**)&partialOutDataBuf, sizeof(float) * signalLength);

  float *outDataBuf;
  hipMalloc((void**)&outDataBuf, sizeof(float) * signalLength);

  std::cout << "Executing kernel for " 
            << iterations << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
  {
    runKernel(inData, hOutData, dOutData, dPartialOutData,
              signalLength, inDataBuf, partialOutDataBuf, outDataBuf);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device offload time " << (time * 1e-9f) / iterations << " (s)\n";

  // Verify
  calApproxFinalOnHost(inData, hOutData, signalLength);

  bool ok = true;
  for(unsigned int i = 0; i < signalLength; ++i)
  {
    if(fabs(dOutData[i] - hOutData[i]) > 0.1f)
    {
      ok = false;
      break;
    }
  }

  hipFree(inDataBuf);
  hipFree(outDataBuf);
  hipFree(partialOutDataBuf);

  free(inData);
  free(dOutData);
  free(dPartialOutData);
  free(hOutData);

  if(ok)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
  if (!ok) exit(1);

  return 0;
}
