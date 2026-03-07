#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <random>
#include <algorithm>
#include <chrono>
#include <climits>
#include "helper_string.h"
#include <cstdlib>

template <typename T, bool floatKeys> bool testSort(int argc, char **argv) {
  int cmdVal;
  int keybits = 32;

  int numElements = 1048576;
  bool keysOnly = checkCmdLineFlag(argc, (const char **)argv, "keysonly");
  bool quiet = checkCmdLineFlag(argc, (const char **)argv, "quiet");

  if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
    cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    numElements = cmdVal;

    if (cmdVal < 0) {
      printf("Error: elements must be > 0, elements=%d is invalid\n", cmdVal);
      exit(EXIT_SUCCESS);
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "keybits")) {
    cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "keybits");
    keybits = cmdVal;

    if (keybits <= 0) {
      printf("Error: keybits must be > 0, keybits=%d is invalid\n", keybits);
      exit(EXIT_SUCCESS);
    }
  }

  unsigned int numIterations = (numElements >= 16777216) ? 10 : 100;

  if (checkCmdLineFlag(argc, (const char **)argv, "iterations")) {
    cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
    numIterations = cmdVal;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printf("Command line:\nradixSortThrust [-option]\n");
    printf("Valid options:\n");
    printf("-n=<N>        : number of elements to sort\n");
    printf("-keybits=bits : keybits must be > 0\n");
    printf(
        "-keysonly     : only sort an array of keys (default sorts key-value "
        "pairs)\n");
    printf(
        "-float        : use 32-bit float keys (default is 32-bit unsigned "
        "int)\n");
    printf(
        "-quiet        : Output only the number of elements and the time to "
        "sort\n");
    printf("-help         : Output a help message\n");
    exit(EXIT_SUCCESS);
  }

  if (!quiet)
    printf("\nSorting %d %d-bit %s keys %s\n\n", numElements, keybits,
           floatKeys ? "float" : "unsigned int",
           keysOnly ? "(only)" : "and values");

  std::vector<T> h_keys(numElements);
  std::vector<T> h_keysSorted(numElements);
  std::vector<unsigned int> h_values;

  if (!keysOnly) h_values = std::vector<unsigned int>(numElements);

  std::mt19937 rng (19937);
  if (floatKeys) {
    std::uniform_real_distribution<float> u01(0, 1);
    for (int i = 0; i < numElements; i++) h_keys[i] = u01(rng);
  } else {
    std::uniform_int_distribution<unsigned int> u(0, UINT_MAX);
    for (int i = 0; i < numElements; i++) h_keys[i] = u(rng);
  }

  if (!keysOnly) 
    for (int i = 0; i < numElements; i++) h_values[i] = i;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Copy data onto the device
  T *d_keys = sycl::malloc_device<T>(numElements, q);
  unsigned int *d_values = sycl::malloc_device<unsigned int>(numElements, q);

  double totalTime = 0;

  auto policy = oneapi::dpl::execution::make_device_policy(q);

  for (unsigned int i = 0; i < numIterations; i++) {
    // reset data before sort
    q.copy(h_keys.data(), d_keys, numElements);

    if (!keysOnly) q.copy(h_values.data(), d_values, numElements);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    if (keysOnly)
      oneapi::dpl::sort(policy, d_keys, d_keys + numElements);
    else
      oneapi::dpl::sort_by_key(policy, d_keys, d_keys + numElements, d_values);

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    totalTime += time;
  }

  totalTime = totalTime * 1e-9f / numIterations;

  printf("Throughput = %.4f MElements/s, Time = %.5lf s\n",
         1.0e-6f * numElements / totalTime, totalTime);

  // Get results back to host for correctness checking
  q.copy(d_keys, h_keysSorted.data(), numElements);

  if (!keysOnly)
    if (!keysOnly) q.copy(d_values, h_values.data(), numElements);

  // Check results
  q.wait();
  sycl::free(d_keys, q);
  sycl::free(d_values, q);

  return oneapi::dpl::is_sorted(oneapi::dpl::execution::seq,
                                h_keysSorted.begin(), h_keysSorted.end());
}

int main(int argc, char **argv) {
  printf("%s Starting...\n\n", argv[0]);

  bool bTestResult = false;

  if (checkCmdLineFlag(argc, (const char **)argv, "float"))
    bTestResult = testSort<float, true>(argc, argv);
  else
    bTestResult = testSort<unsigned int, false>(argc, argv);

  printf(bTestResult ? "PASS\n" : "FAIL\n");
  if (!bTestResult) exit(1);
}
