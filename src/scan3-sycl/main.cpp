#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <chrono>
#include <sycl/sycl.hpp>
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Create input buffer on device
  sycl::buffer<float, 1> inputBuffer (input, length);

  // Create output buffer on device
  sycl::buffer<float, 1> outputBuffer (length);

  std::cout << "Executing kernel for " << iterations << " iterations\n";
  std::cout << "-------------------------------------------\n";

  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto ibuf_beg = oneapi::dpl::begin(inputBuffer, sycl::read_only);
  auto ibuf_end = oneapi::dpl::end(inputBuffer, sycl::read_only);
  auto obuf_beg = oneapi::dpl::begin(outputBuffer, sycl::write_only, sycl::no_init);

  // warmup
  oneapi::dpl::exclusive_scan(policy, ibuf_beg, ibuf_end, obuf_beg, 0.f);

  auto start = std::chrono::steady_clock::now();

  for(int n = 0; n < iterations; n++)
  {
    oneapi::dpl::exclusive_scan(policy, ibuf_beg, ibuf_end, obuf_beg, 0.f);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time of oneDPL exclusive scan: "
            << time * 1e-3f / iterations << " (us)\n";

  q.submit([&] (sycl::handler &cgh) {
    auto acc = outputBuffer.get_access<sycl::access::mode::read>(cgh);
    cgh.copy(acc, output);
  }).wait();

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
  if (!(compare<float>(output, verificationOutput, length, (float)0.001))) exit(1);

  free(input);
  free(output);
  free(verificationOutput);
  return 0;
}
