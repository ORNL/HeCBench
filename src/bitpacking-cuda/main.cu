#include "utils.h"
#include <cstdlib>

template <typename T>
void toGPU(T* const output, T const* const input, size_t const num)
{
  CUDA_RT_CALL(
      cudaMemcpy(output, input, num * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void fromGPU(T* const output, T const* const input, size_t const num)
{
  CUDA_RT_CALL(
      cudaMemcpy(output, input, num * sizeof(T), cudaMemcpyDeviceToHost));
}

template <>
void fromGPU<void>(void* const output, void const* const input, size_t const num)
{
  CUDA_RT_CALL(cudaMemcpy(output, input, num, cudaMemcpyDeviceToHost));
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
  T* input;

  CUDA_RT_CALL(cudaMalloc((void**)&input, n * sizeof(T)));
  toGPU(input, inputHost, n);

  void* output;
  size_t const packedSize = (((numBitsMax * n) / 64U) + 1U) * 8U;

  size_t* numDevice;
  CUDA_RT_CALL(cudaMalloc((void**)&numDevice, sizeof(numDevice)));
  toGPU(numDevice, &n, 1);

  CUDA_RT_CALL(cudaMalloc(&output, packedSize));
  CUDA_RT_CALL(cudaMemset(output, 0, packedSize));

  T* minValueDevice;
  CUDA_RT_CALL(cudaMalloc((void**)&minValueDevice, sizeof(*minValueDevice)));
  unsigned char* numBitsDevice;
  CUDA_RT_CALL(cudaMalloc((void**)&numBitsDevice, sizeof(*numBitsDevice)));

  void* workspace;
  size_t workspaceBytes = requiredWorkspaceSize(n, TypeOf<T>());
  CUDA_RT_CALL(cudaMalloc(&workspace, workspaceBytes));

  const nvcompType_t inType = TypeOf<T>();

  compress(
      workspace,
      workspaceBytes,
      inType,
      output,
      input,
      numDevice,
      n,
      minValueDevice,
      numBitsDevice);

  fromGPU(minValOut, minValueDevice, 1);

  unsigned char numBits;
  fromGPU(&numBits, numBitsDevice, 1);
  *numBitsOut = numBits;

  fromGPU(outputHost, output, std::min(packedSize, n * sizeof(T)));

  CUDA_RT_CALL(cudaFree(input));
  CUDA_RT_CALL(cudaFree(output));
  CUDA_RT_CALL(cudaFree(workspace));
  CUDA_RT_CALL(cudaFree(minValueDevice));
  CUDA_RT_CALL(cudaFree(numBitsDevice));
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
