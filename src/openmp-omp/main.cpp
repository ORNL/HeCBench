/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "reference.h"

int main(int argc, char *argv[]) {

  printf("%s Starting...\n\n", argv[0]);
  const int repeat = atoi(argv[1]);

  int num_gpus = 1; // Use a single GPU

  printf("number of host CPUs:\t%d\n", omp_get_num_procs());
  printf("number of devices:\t%d\n", num_gpus);

  // initialize data
  unsigned int nwords = num_gpus * 33554432;
  unsigned int nbytes = nwords * sizeof(int);
  int b = 3;   // value by which the array is incremented
  int *a = (int *)malloc(nbytes); // pointer to data on the CPU

  if (NULL == a) {
    printf("couldn't allocate CPU memory\n");
    return 1;
  }
  double overhead; // record overhead of first run

  //   Recall that all variables declared inside an "omp parallel" scope are
  //   local to each CPU thread
  for (int i = 0; i < 2; i++) {
    for (int f = 1; f <= 32; f = f*2) {
      double start = omp_get_wtime();
      omp_set_num_threads(f * num_gpus); 
      #pragma omp parallel
      {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        // pointer to this CPU thread's portion of data
        unsigned int nwords_per_kernel = nwords / num_cpu_threads;
        int *sub_a = a + cpu_thread_id * nwords_per_kernel;

        for (unsigned int n = 0; n < nwords_per_kernel; n++)
          sub_a[n] = n + cpu_thread_id * nwords_per_kernel;

        #pragma omp target data map (tofrom: sub_a[0:nwords_per_kernel])
        {
          #pragma omp target teams distribute parallel for thread_limit(256)
          for (int idx = 0; idx < nwords_per_kernel; idx++) {
            for (int i = 0; i < repeat; i++)
              sub_a[idx] += i % b;
          }
        }
      }
      double end = omp_get_wtime();
      printf("Work took %f seconds with %d CPU threads\n", end - start, f*num_gpus);
      
      if (f == 1) {
        if (i == 0) 
          overhead = end - start;
        else
          overhead = overhead - (end - start);
      }
      // check the result
      bool bResult = correctResult(a, nwords, b, repeat);
      printf("%s\n", bResult ? "PASS" : "FAIL");
      if (!bResult) exit(1);
    }
  }
  printf("Runtime overhead of first run is %f seconds\n", overhead);

  free(a);
  return 0;
}
