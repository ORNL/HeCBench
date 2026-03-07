#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <hip/hip_runtime.h>

#include "kernels.cu"

void runDevice(float* input, float* output, int n, int repeat)
{
  float* d_answer;
  hipMalloc(&d_answer, 21 * sizeof(float) * n);

  float* d_input;
  hipMalloc(&d_input, 9 * sizeof(float) * n);
  hipMemcpy(d_input, input, 9 * sizeof(float) * n, hipMemcpyHostToDevice);

  int threads = 256;
  int pblks = int(n / threads) + 1;

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    hipLaunchKernelGGL(svd3_SOA, pblks, threads , 0, 0, d_input, d_answer, n);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  hipMemcpy(output, d_answer, 21 * sizeof(float) * n, hipMemcpyDeviceToHost);
  hipFree(d_answer);
  hipFree(d_input);
}

void svd3x3_ref(float* input, float* output, int testsize)
{
  for (int tid = 0; tid < testsize; tid++) 
    svd(
      input[tid + 0 * testsize], input[tid + 1 * testsize], input[tid + 2 * testsize],
      input[tid + 3 * testsize], input[tid + 4 * testsize], input[tid + 5 * testsize],
      input[tid + 6 * testsize], input[tid + 7 * testsize], input[tid + 8 * testsize],

      output[tid + 0 * testsize], output[tid + 1 * testsize], output[tid + 2 * testsize],
      output[tid + 3 * testsize], output[tid + 4 * testsize], output[tid + 5 * testsize],
      output[tid + 6 * testsize], output[tid + 7 * testsize], output[tid + 8 * testsize],
      output[tid + 9 * testsize], output[tid + 10 * testsize], output[tid + 11 * testsize],
      output[tid + 12 * testsize], output[tid + 13 * testsize], output[tid + 14 * testsize],
      output[tid + 15 * testsize], output[tid + 16 * testsize], output[tid + 17 * testsize],
      output[tid + 18 * testsize], output[tid + 19 * testsize], output[tid + 20 * testsize]
    );
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path to file> <repeat>\n";
    return 1;
  }

  // Load data
  const char* filename = argv[1];
  const int repeat = atoi(argv[2]);

  std::ifstream myfile (filename);
  if (!myfile.is_open()) {
    std::cout << "ERROR: failed to open " << filename << std::endl;
    return -1;
  }

  int testsSize;
  myfile >> testsSize;
  std::cout << "dataset size: " << testsSize << std::endl;
  if (testsSize <= 0) {
    std::cout << "ERROR: invalid dataset size\n";
    return -1;
  }

  float* input = (float*)malloc(sizeof(float) * 9 * testsSize);

  // host and device results
  float* result = (float*)malloc(sizeof(float) * 21 * testsSize);
  float* result_h = (float*)malloc(sizeof(float) * 21 * testsSize);

  int count = 0;
  for (int i = 0; i < testsSize; i++)
    for (int j = 0; j < 9; j++) myfile >> input[count++];
  myfile.close();

  // SVD 3x3 on a GPU 
  runDevice(input, result, testsSize, repeat);

  // run CPU 3x3 to verify results
  bool ok = true;
  svd3x3_ref(input, result_h, testsSize);

  for (int i = 0; i < testsSize; i++)
  {
    if (fabsf(result[i] - result_h[i]) > 1e-3) {
      std::cout << result[i] << " " << result_h[i] << std::endl;
      ok = false;
      break;
    }
  }

  if (ok)
    std::cout << "PASS\n";
  else
    std::cout << "FAIL\n";

  free(input);
  free(result);
  free(result_h);
  if (!ok) return 1;
  return 0;
}
