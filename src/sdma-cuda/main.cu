#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda.h>

#define CHECK(cmd)                                                         \
  do {                                                                     \
    cudaError_t err = (cmd);                                               \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA error " << #cmd << ": " << cudaGetErrorString(err)\
                << std::endl;                                              \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

static void fill_pattern(uint8_t* buf, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    buf[i] = static_cast<uint8_t>(i);
  }
}

static bool verify(const uint8_t* got, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint8_t expected = static_cast<uint8_t>(i);
    if (got[i] != expected) {
      std::cerr << "Verification failed at offset " << i << ": got "
                << static_cast<unsigned>(got[i]) << " expected "
                << static_cast<unsigned>(expected) << std::endl;
      return false;
    }
  }
  return true;
}

static void report(const char* kind, size_t bytes, long long ns, int repeat) {
  double avg_us = (static_cast<double>(ns) * 1e-3) / repeat;
  double seconds = static_cast<double>(ns) * 1e-9 / repeat;
  double gbps = seconds > 0.0 ? static_cast<double>(bytes) / seconds / 1e9 : 0.0;
  std::cout << "Copy " << bytes << " bytes with " << kind << " takes " << avg_us
            << " us (" << gbps << " GB/s)" << std::endl;
}

// Host time for `repeat` ordered copies, after one untimed warmup copy.
static long long time_copy(uint8_t* dst, uint8_t* src, size_t bytes, int repeat) {
  CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, 0));
  CHECK(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, 0));
  }
  CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <repeat>\n";
    return 1;
  }
  const int repeat = std::atoi(argv[1]);
  if (repeat <= 0) {
    std::cerr << "repeat must be positive\n";
    return 1;
  }

  int device_count = 0;
  CHECK(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::cerr << "No CUDA devices available\n";
    return 2;
  }
  CHECK(cudaSetDevice(0));

  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "Device: " << prop.name
            << ", asyncEngineCount=" << prop.asyncEngineCount << std::endl;
  std::cout << "CUDA has no SDMA copy kind. Timing std::chrono around "
            << "cudaMemcpyDeviceToDevice." << std::endl;

  // Keep this range identical in sdma-hip and sdma-sycl. It does not follow
  // HIP's GPU_FORCE_BLIT_COPY_SIZE.
  constexpr int kMinShift = 8;   // 256 B
  constexpr int kMaxShift = 30;  // 1 GiB
  std::cout << "Sweep is powers of two from " << (size_t{1} << kMinShift)
            << " bytes to " << (size_t{1} << kMaxShift) << " bytes." << std::endl;

  bool ok = true;
  for (int shift = kMinShift; shift <= kMaxShift; ++shift) {
    const size_t bytes = size_t{1} << shift;
    std::vector<uint8_t> h_src(bytes);
    std::vector<uint8_t> h_dst(bytes);
    fill_pattern(h_src.data(), bytes);

    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_src), bytes));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dst), bytes));
    CHECK(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_dst, 0, bytes));

    long long ns = time_copy(d_dst, d_src, bytes, repeat);
    report("cudaMemcpyDeviceToDevice", bytes, ns, repeat);
    CHECK(cudaMemcpy(h_dst.data(), d_dst, bytes, cudaMemcpyDeviceToHost));
    ok = verify(h_dst.data(), bytes) && ok;

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
  }

  std::cout << (ok ? "PASS" : "FAIL") << std::endl;
  return ok ? 0 : 1;
}
