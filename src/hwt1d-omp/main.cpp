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

  std::cout << "Executing kernel for " 
            << iterations << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

#pragma omp target data map(alloc: inData[0:signalLength], \
                                   dOutData[0:signalLength], \
                                   dPartialOutData[0:signalLength])
{
  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; i++)
  {
    unsigned int levels = 0;

    getLevels(signalLength, &levels);  // no return from OMP region

    unsigned int actualLevels = levels;

    //max levels on device should be decided by kernelWorkGroupSize
    //int tempVar = (int)(log((float)kernelInfo.kernelWorkGroupSize) / log((float)2));
    //int maxLevelsOnDevice = tempVar + 1;
    // Assume work group size is 256
    const int maxLevelsOnDevice = 9;

    float* temp = (float*)malloc(signalLength * sizeof(float));
    memcpy(temp, inData, signalLength * sizeof(float));

    int levelsDone = 0;
    int one = 1;
    unsigned int curLevels = 0;
    unsigned int curSignalLength;
    while((unsigned int)levelsDone < actualLevels)
    {
      curLevels = (levels < maxLevelsOnDevice) ? levels : maxLevelsOnDevice;

      // Set the signal length for current iteration
      if(levelsDone == 0)
      {
        curSignalLength = signalLength;
      }
      else
      {
        curSignalLength = (one << levels);
      }

      // Set group size
      unsigned int groupSize = (1 << curLevels) / 2;

      unsigned int totalLevels = levels;

      #pragma omp target update to(inData[0:signalLength])

      const int teams = (curSignalLength >> 1) / groupSize;

      #pragma omp target teams num_teams(teams) thread_limit(groupSize)
      {
        float lmem [512];
        #pragma omp parallel 
        {
          size_t localId = omp_get_thread_num();
          size_t groupId = omp_get_team_num();
          size_t localSize = omp_get_num_threads();
          
          /**
           * Read input signal data from global memory
           * to shared memory
           */
          float t0 = inData[groupId * localSize * 2 + localId];
          float t1 = inData[groupId * localSize * 2 + localSize + localId];
          // Divide with signal length for normalized decomposition
          if(0 == levelsDone)
          {
             float r = 1.f / sqrtf((float)curSignalLength);
             t0 *= r;
             t1 *= r;
          }
          lmem[localId] = t0;
          lmem[localSize + localId] = t1;
           
          #pragma omp barrier
          
          unsigned int levels = totalLevels > maxLevelsOnDevice ? maxLevelsOnDevice: totalLevels;
          unsigned int activeThreads = (1 << levels) / 2;
          unsigned int midOutPos = curSignalLength / 2;
          
          const float rsqrt_two = 0.7071f;
          for(unsigned int i = 0; i < levels; ++i)
          {

              float data0, data1;
              if(localId < activeThreads)
              {
                  data0 = lmem[2 * localId];
                  data1 = lmem[2 * localId + 1];
              }

              /* make sure all work items have read from lmem before modifying it */
              #pragma omp barrier

              if(localId < activeThreads)
              {
                  lmem[localId] = (data0 + data1) * rsqrt_two;
                  unsigned int globalPos = midOutPos + groupId * activeThreads + localId;
                  dOutData[globalPos] = (data0 - data1) * rsqrt_two;
             
                  midOutPos >>= 1;
              }
              activeThreads >>= 1;
              #pragma omp barrier
          }
      
          /**
           * Write 0th element for the next decomposition
           * steps which are performed on host 
           */
          
           if(0 == localId)
              dPartialOutData[groupId] = lmem[0];
        }
      }

      #pragma omp target update from(dOutData[0:signalLength])
      #pragma omp target update from(dPartialOutData[0:signalLength])

      if(levels <= maxLevelsOnDevice)
      {
        dOutData[0] = dPartialOutData[0];
        memcpy(hOutData, dOutData, (one << curLevels) * sizeof(float));
        memcpy(dOutData + (one << curLevels), hOutData + (one << curLevels),
            (signalLength  - (one << curLevels)) * sizeof(float));
        break;
      }
      else
      {
        levels -= maxLevelsOnDevice;
        memcpy(hOutData, dOutData, curSignalLength * sizeof(float));
        memcpy(inData, dPartialOutData, (one << levels) * sizeof(float));
        levelsDone += (int)maxLevelsOnDevice;
      }
    }

    memcpy(inData, temp, signalLength * sizeof(float));
    free(temp);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average device offload time " << (time * 1e-9f) / iterations << " (s)\n";
}

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
