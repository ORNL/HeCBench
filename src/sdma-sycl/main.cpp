#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

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
static long long time_copy(sycl::queue& q, uint8_t* dst, uint8_t* src, size_t bytes,
                           int repeat) {
  q.memcpy(dst, src, bytes);
  q.wait();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    q.memcpy(dst, src, bytes);
  }
  q.wait();
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

  try {
#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "SYCL has no SDMA copy kind. Timing std::chrono around "
              << "queue::memcpy." << std::endl;

    // Keep this range identical in sdma-hip and sdma-cuda. It does not follow
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

      uint8_t* d_src = sycl::malloc_device<uint8_t>(bytes, q);
      uint8_t* d_dst = sycl::malloc_device<uint8_t>(bytes, q);
      if (d_src == nullptr || d_dst == nullptr) {
        std::cerr << "Device memory allocation failed\n";
        sycl::free(d_src, q);
        sycl::free(d_dst, q);
        return 1;
      }

      q.memcpy(d_src, h_src.data(), bytes);
      q.memset(d_dst, 0, bytes);
      q.wait();

      long long ns = time_copy(q, d_dst, d_src, bytes, repeat);
      report("queue::memcpy", bytes, ns, repeat);
      q.memcpy(h_dst.data(), d_dst, bytes).wait();
      ok = verify(h_dst.data(), bytes) && ok;

      sycl::free(d_src, q);
      sycl::free(d_dst, q);
    }

    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
    return 1;
  }
}
