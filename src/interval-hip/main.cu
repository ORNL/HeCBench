/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

/* Example of program using the interval_gpu<T> template class and operators:
 * Search for roots of a function using an interval Newton method.
  *
 * 0: the first implementation
 * 1: the second implementation
 *
 */

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <hip/hip_runtime.h>
#include "interval.h"
#include "gpu_interval.h"
#include "cpu_interval.h"
#include <cstdlib>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <implementation choice> <repeat>\n", argv[0]);
    return 1;
  }

  const int implementation_choice = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  switch (implementation_choice) {
    case 0:
      printf("GPU implementation 1\n");
      break;

    case 1:
      printf("GPU implementation 2\n");
      break;

    default:
      printf("GPU implementation 1\n");
  }

  interval_gpu<T> *d_result;
  int *d_nresults;
  int *h_nresults = new int[THREADS];

  hipMalloc((void **)&d_result, THREADS * DEPTH_RESULT * sizeof(*d_result));
  hipMalloc((void **)&d_nresults, THREADS * sizeof(*d_nresults));

  interval_gpu<T> i(0.01f, 4.0f);
  std::cout << "Searching for roots in [" << i.lower() << ", " << i.upper()
            << "]...\n";

  auto start = std::chrono::steady_clock::now();

  for (int it = 0; it < repeat; ++it) {
    test_interval_newton<T><<<GRID_SIZE, BLOCK_SIZE>>>(
      d_result, d_nresults, i, implementation_choice);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  I_CPU *h_result = new I_CPU[THREADS * DEPTH_RESULT];
  hipMemcpy(h_result, d_result, THREADS * DEPTH_RESULT * sizeof(*d_result),
             hipMemcpyDeviceToHost);

  hipMemcpy(h_nresults, d_nresults, THREADS * sizeof(*d_nresults),
             hipMemcpyDeviceToHost);

  std::cout << "Found " << h_nresults[0]
            << " intervals that may contain the root(s)\n";
  std::cout.precision(15);

  for (int i = 0; i != h_nresults[0]; ++i) {
    std::cout << " i[" << i << "] ="
              << " [" << h_result[THREADS * i + 0].lower() << ", "
              << h_result[THREADS * i + 0].upper() << "]\n";
  }

  std::cout << "Number of equations solved: " << THREADS << "\n";
  std::cout << "Average execution time of test_interval_newton: "
            << (time * 1e-3f) / repeat << " us\n";

  hipFree(d_result);
  hipFree(d_nresults);

  // Compute the results using a CPU implementation based on the Boost library
  I_CPU i_cpu(0.01f, 4.0f);
  I_CPU *h_result_cpu = new I_CPU[THREADS * DEPTH_RESULT];
  int *h_nresults_cpu = new int[THREADS];
  test_interval_newton_cpu<I_CPU>(h_result_cpu, h_nresults_cpu, i_cpu);

  // Compare the CPU and GPU results
  bool bTestResult =
      checkAgainstHost(h_nresults, h_nresults_cpu, h_result, h_result_cpu);
  std::cout << (bTestResult ? "PASS" : "FAIL") << "\n";

  delete[] h_result_cpu;
  delete[] h_nresults_cpu;
  delete[] h_result;
  delete[] h_nresults;

  if (!bTestResult) return 1;
  return 0;
}
