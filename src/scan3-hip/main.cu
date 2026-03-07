#include <chrono>
#include <hip/hip_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <hipcub/hipcub.hpp>
#include "scan.h"
#include <cstdlib>

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
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <repeat> <input length>\n";
    return 1;
  }
  int iterations = atoi(argv[1]);
  int length = atoi(argv[2]);

  if(iterations < 1)
  {
    std::cout << "Error, iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }
  if(!isPowerOf2(length))
  {
    length = roundToPowerOf2(length);
  }

  // input buffer size
  unsigned int sizeBytes = length * sizeof(float);

  float* input = (float*) malloc (sizeBytes);

  // store device results for verification
  float* output = (float*) malloc (sizeBytes);

  // random initialisation of input
  fillRandom<float>(input, length, 1, 0, 255);

  // verification
  float* verificationOutput = (float*)malloc(sizeBytes);
  memset(verificationOutput, 0, sizeBytes);

  // reference implementation
  scanLargeArraysCPUReference(verificationOutput, input, length);

  // Create input buffer on device
  float* inputBuffer;
  hipMalloc((void**)&inputBuffer, sizeBytes);
  hipMemcpy(inputBuffer, input, sizeBytes, hipMemcpyHostToDevice);

  // Create output buffer on device
  float* outputBuffer;
  hipMalloc((void**)&outputBuffer, sizeBytes);

  thrust::device_ptr<float> ibuf(inputBuffer);
  thrust::device_ptr<float> obuf(outputBuffer);

  std::cout << "Executing kernel for " << iterations << " iterations\n";
  std::cout << "-------------------------------------------\n";

  // warmup
  thrust::exclusive_scan(ibuf, ibuf + length, obuf, 0.f);

  auto start = std::chrono::steady_clock::now();

  for(int n = 0; n < iterations; n++)
  {
    thrust::exclusive_scan(ibuf, ibuf + length, obuf, 0.f);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of HIP Thrust exclusive scan: "
            << time * 1e-3f / iterations << " (us)\n";

  hipMemcpy(output, outputBuffer, sizeBytes, hipMemcpyDeviceToHost);

  // compare the results and see if they match
  if (compare<float>(output, verificationOutput, length, (float)0.001))
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
  if (!(compare<float>(output, verificationOutput, length, (float)0.001))) exit(1);

  // include the overhead of allocating temporary device storage
  start = std::chrono::steady_clock::now();

  for(int n = 0; n < iterations; n++)
  {
    // Determine temporary device storage requirements
    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    hipcub::DeviceScan::ExclusiveScan(
      d_temp_storage, temp_storage_bytes,
      inputBuffer, outputBuffer, hipcub::Sum(), 0.f, length);
    
    // Allocate temporary storage
    hipMalloc(&d_temp_storage, temp_storage_bytes);
    
    hipcub::DeviceScan::ExclusiveScan(
      d_temp_storage, temp_storage_bytes,
      inputBuffer, outputBuffer, hipcub::Sum(), 0.f, length);

    hipFree(d_temp_storage);
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of HIP CUB exclusive scan: "
            << time * 1e-3f / iterations << " (us)\n";

  hipMemcpy(output, outputBuffer, sizeBytes, hipMemcpyDeviceToHost);

  // compare the results and see if they match
  if (compare<float>(output, verificationOutput, length, (float)0.001))
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
  if (!(compare<float>(output, verificationOutput, length, (float)0.001))) exit(1);

  hipFree(inputBuffer);
  hipFree(outputBuffer);

  free(input);
  free(output);
  free(verificationOutput);
  return 0;
}
