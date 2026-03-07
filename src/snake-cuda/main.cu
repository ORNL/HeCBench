/*
 * Copyright (c) <2017 - 2020>, ETH Zurich and Bilkent University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this list
 * of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice, this
 *   list of conditions and the following disclaimer in the documentation and/or other
 *   materials provided with the distribution.
 * - Neither the names of the ETH Zurich, Bilkent University,
 *   nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without specific
 *   prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <cuda.h>

using namespace std::chrono;

#define warp_size 32
#define NBytes 8

__host__ __device__
inline uint lsr(uint x, int sa) {
  if(sa > 0 && sa < 32) return (x >> sa);
  return x;
}

__host__ __device__
inline uint lsl(uint x, int sa) {
  if (sa > 0 && sa < 32) return (x << sa);
  return x;
}

__host__ __device__
inline uint set_bit(uint &data, int y) {
  data |= lsl(1, y);
  return data;
}

#include "kernel.h"
#include "reference.h"

int main(int argc, const char * const argv[])
{
  if (argc != 5) {
    printf("Usage: ./%s [ReadLength] [ReadandRefFile] [#reads] [repeat]\n", argv[0]);
    exit(-1);
  }

  int ReadLength = atoi(argv[1]);//in my inputs, it is always 100. Just for the generality we keep it as a variable
  int NumReads = atoi(argv[3]); // Number of reads
  int repeat = atoi(argv[4]);
  int Size_of_int_in_Bit = 32; //in Bits 

  FILE * fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;
  char *p;//when reading each char_basepair from the file, we read it into the p.

  int Number_of_warps_inside_each_block = 8; 
  int Concurrent_threads_In_Block = warp_size * Number_of_warps_inside_each_block;
  int Number_of_blocks_inside_each_kernel = (NumReads + Concurrent_threads_In_Block - 1) / 
                                            Concurrent_threads_In_Block;

  int F_ErrorThreshold =0;

  uint* ReadSeq = (uint *) calloc(NumReads * 8, sizeof(uint));
  uint* RefSeq = (uint *) calloc(NumReads * 8, sizeof(uint));
  int* DFinal_Results = (int *) calloc(NumReads, sizeof(int));
  int* HFinal_Results = (int *) calloc(NumReads, sizeof(int));

  int tokenIndex=1;
  fp = fopen(argv[2], "r");
  if (!fp){
    printf("The file %s does not exist or you do not have access permission\n", argv[2]);
    return 0;
  }
  for(int this_read = 0; this_read < NumReads; this_read++) {
    read = getline(&line, &len, fp);
    tokenIndex=1;
    for (p = strtok(line, "\t"); p != NULL; p = strtok(NULL, "\t"))
    {
      if (tokenIndex==1)
      {
        for (int j = 0; j < ReadLength; j++)
        {
          if(p[j] == 'A')
          {
            //do nothing (this is like storing 00)
          }
          else if (p[j] == 'C')
          {
            ReadSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)] = set_bit(ReadSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_int_in_Bit/2)) * 2 + 1));
          }
          else if (p[j] == 'G')
          {
            ReadSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)] = set_bit(ReadSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_int_in_Bit/2)) * 2));
          }
          else if (p[j] == 'T')
          {
            ReadSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)] = set_bit(ReadSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_int_in_Bit/2)) * 2));

            ReadSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)] = set_bit(ReadSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_int_in_Bit/2)) * 2 + 1));
          }
        }
      }
      else if(tokenIndex==2)
      {
        for (int j = 0; j < ReadLength; j++)
        {
          if(p[j] == 'A')
          {
            //do nothing (this is like storing 00)
          }
          else if (p[j] == 'C')
          {
            RefSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)] = set_bit(RefSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_int_in_Bit/2)) * 2 + 1));
          }
          else if (p[j] == 'G')
          {
            RefSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)] = set_bit(RefSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_int_in_Bit/2)) * 2));
          }
          else if (p[j] == 'T')
          {
            RefSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)] = set_bit(RefSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_int_in_Bit/2)) * 2));

            RefSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)] = set_bit(RefSeq[((j*2/Size_of_int_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_int_in_Bit/2)) * 2 + 1));
          }
        }
      }
      tokenIndex=tokenIndex+1;
    }
  }
  fclose(fp);

  uint* Dev_ReadSeq;
  uint* Dev_RefSeq;
  int* Dev_Results;
  cudaMalloc((void**)&Dev_ReadSeq, sizeof(uint) * NumReads * 8);
  cudaMalloc((void**)&Dev_RefSeq, sizeof(uint) * NumReads * 8);
  cudaMalloc((void**)&Dev_Results, sizeof(int) * NumReads);

  dim3 grid (Number_of_blocks_inside_each_kernel);
  dim3 block (Concurrent_threads_In_Block);

  cudaMemcpy(Dev_ReadSeq, ReadSeq, sizeof(int) * NumReads * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(Dev_RefSeq, RefSeq, sizeof(int) * NumReads * 8, cudaMemcpyHostToDevice);

  bool error = false;
  for (int loopPar = 0; loopPar <= 25; loopPar++) {

    F_ErrorThreshold = (loopPar*ReadLength)/100;

    auto t1 = high_resolution_clock::now();

    for (int n = 0; n < repeat; n++) {
      sneaky_snake<<<grid, block>>>(Dev_ReadSeq, Dev_RefSeq, Dev_Results, NumReads, F_ErrorThreshold);
    }

    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();
    double elapsed_time = duration_cast<microseconds>(t2 - t1).count();
    
    cudaMemcpy(DFinal_Results, Dev_Results, sizeof(int) * NumReads, cudaMemcpyDeviceToHost);

    // verify
    sneaky_snake_ref(ReadSeq, RefSeq, HFinal_Results, NumReads, F_ErrorThreshold);
    error = memcmp(DFinal_Results, HFinal_Results, NumReads * sizeof(int));
    if (error) break;

    // stats
    int D_accepted = 0;
    for(int i = 0; i < NumReads; i++) if(DFinal_Results[i] == 1) D_accepted++;

    printf("Error threshold: %2d | Average kernel time (us): %5.4f | Accepted: %10d | Rejected: %10d\n", 
          F_ErrorThreshold, elapsed_time / repeat, D_accepted, NumReads - D_accepted);
  }
  printf("%s\n", error ? "FAIL" : "PASS");
  
  free(ReadSeq);
  free(RefSeq);
  free(DFinal_Results);
  free(HFinal_Results);
  cudaFree(Dev_ReadSeq);
  cudaFree(Dev_RefSeq);
  cudaFree(Dev_Results);
  if (error) return 1;
  return 0;
}
