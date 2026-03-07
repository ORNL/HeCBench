#include "utils.h"
#include <cstdlib>

template <typename T>
void toGPU(sycl::queue &q, T* const output, T const* const input, size_t const num)
{
  q.memcpy(output, input, num * sizeof(T)).wait();
}

template <typename T>
void fromGPU(sycl::queue &q, T* const output, T const* const input, size_t const num)
{
  q.memcpy(output, input, num * sizeof(T)).wait();
}

template <>
void fromGPU<void>(sycl::queue &q, void* const output, void const* const input, size_t const num)
{
  q.memcpy(output, input, num).wait();
}

template <typename T>
void runBitPackingOnGPU(
    T const* const inputHost,
    void* const outputHost,
    int const numBitsMax,
    size_t const n,
    int* const numBitsOut,
    T* const minValOut)
{

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  T* input;

  input = (T *)sycl::malloc_device(n * sizeof(T), q);
  toGPU(q, input, inputHost, n);

  void* output;
  size_t const packedSize = (((numBitsMax * n) / 64U) + 1U) * 8U;

  size_t* numDevice;
  numDevice = (size_t *)sycl::malloc_device(sizeof(numDevice), q);
  toGPU(q, numDevice, &n, 1);

  output = (void *)sycl::malloc_device(packedSize, q);
  q.memset(output, 0, packedSize).wait();

  T* minValueDevice;
  minValueDevice = (T *)sycl::malloc_device(sizeof(T), q);

  unsigned char* numBitsDevice;
  numBitsDevice = (unsigned char *)sycl::malloc_device(sizeof(unsigned char), q);

  void* workspace;
  size_t workspaceBytes = requiredWorkspaceSize(n, TypeOf<T>());
  workspace = (void *)sycl::malloc_device(workspaceBytes, q);

  const nvcompType_t inType = TypeOf<T>();

  compress(
      q,
      workspace,
      workspaceBytes,
      inType,
      output,
      input,
      numDevice,
      n,
      minValueDevice,
      numBitsDevice);

  fromGPU(q, minValOut, minValueDevice, 1);

  unsigned char numBits;
  fromGPU(q, &numBits, numBitsDevice, 1);
  *numBitsOut = numBits;

  fromGPU(q, outputHost, output, std::min(packedSize, n * sizeof(T)));

  sycl::free(input, q);
  sycl::free(output, q);
  sycl::free(workspace, q);
  sycl::free(minValueDevice, q);
  sycl::free(numBitsDevice, q);
}

int main() {
  int const offset = 87231;
  int const numBits = 13;

  // unpack doesn't handle 0 bits
  std::vector<size_t> const sizes{2, 123, 3411, 83621, 872163, 100000001};

  using T = int32_t;

  // generate a variety of random numbers
  std::vector<T> source(sizes.back());
  std::srand(0);
  for (T& v : source) {
    v = std::abs(static_cast<T>(std::rand())) % std::numeric_limits<T>::max();
  }

  size_t const numBytes = sizes.back() * sizeof(T);
  T* inputHost = (T*) aligned_alloc (1024, numBytes);
  void* outputHost = aligned_alloc (1024, numBytes);

  for (size_t const n : sizes) {
    for (size_t i = 0; i < n; ++i) {
      inputHost[i] = (source[i] & ((1U << numBits) - 1)) + offset;
    }

    T minValue;
    int numBitsAct;

    printf("Size = %10zu\n", n);
    auto start = std::chrono::steady_clock::now();

    runBitPackingOnGPU(
        inputHost, outputHost, numBits, n, &numBitsAct, &minValue);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Device offload time = %f (s)\n", time * 1e-9f);

    assert(numBitsAct <= numBits);

    // unpack
    std::vector<T> unpackedHost;
    for (size_t i = 0; i < n; ++i) {
      unpackedHost.emplace_back(unpackBytes(
            outputHost, static_cast<uint8_t>(numBitsAct), minValue, i));
    }

    // verify
    assert(unpackedHost.size() == n);

    bool ok = true;
    // checking 100 million entries can take a while, so sample instead
    size_t const numSamples = static_cast<size_t>(std::sqrt(n)) + 1;
    for (size_t i = 0; i < numSamples; ++i) {
      // only works for arrays less than 4 Billion.
      size_t const idx = static_cast<uint32_t>(source[i]) % n;
      if (unpackedHost[idx] != inputHost[idx]) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");  
    if (!ok) exit(1);
  }

  free(inputHost);
  free(outputHost);
}
