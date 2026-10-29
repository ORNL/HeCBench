#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

// hipMemcpyDeviceToDeviceNoCU was added in HIP 6.1. Older headers have no
// enumerator, so those builds time hipMemcpyDeviceToDevice only.
#ifndef SDMA_HAS_NOCU
#if (HIP_VERSION_MAJOR > 6) || \
    (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 1)
#define SDMA_HAS_NOCU 1
#else
#define SDMA_HAS_NOCU 0
#endif
#endif

#define CHECK(cmd)                                                         \
  do {                                                                     \
    hipError_t err = (cmd);                                                \
    if (err != hipSuccess) {                                               \
      std::cerr << "HIP error " << #cmd << ": " << hipGetErrorString(err)  \
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

static void report(const char* kind, const char* note, size_t bytes, long long ns,
                   int repeat) {
  double avg_us = (static_cast<double>(ns) * 1e-3) / repeat;
  double seconds = static_cast<double>(ns) * 1e-9 / repeat;
  double gbps = seconds > 0.0 ? static_cast<double>(bytes) / seconds / 1e9 : 0.0;
  std::cout << "Copy " << bytes << " bytes with " << kind << " (" << note << ") takes "
            << avg_us << " us (" << gbps << " GB/s)" << std::endl;
}

// HIP flag GPU_FORCE_BLIT_COPY_SIZE, in KiB. The default in flags.hpp is 16.
// OpenCL replaces an unset value with 0; HIP keeps 16. rocclr stores
// sdmaCopyThreshold_ = GPU_FORCE_BLIT_COPY_SIZE * 1024.
//
// KernelBlitManager::useShaderCopyBufferPath uses that threshold when the
// caller did not request SDMA and the copy is not already forced to a shader
// blit. Same-device hipMalloc memory is not host-direct-access, so
// hipMemcpyDeviceToDevice takes the shader blit at every size.
// hipMemcpyDeviceToDeviceNoCU is hipCopyBufferSDMA
// (CopyEnginePreference::SDMA): SDMA at every size, with no shader fallback.
// The threshold does select the engine for a transfer that has a
// host-accessible buffer: size <= sdmaCopyThreshold_ uses a shader blit.
//
// The sweep is a fixed power-of-two range and does not move when
// GPU_FORCE_BLIT_COPY_SIZE changes. Keep kMinShift/kMaxShift identical in
// sdma-cuda and sdma-sycl.
static constexpr unsigned long long kDefaultBlitKib = 16;
static constexpr int kMinShift = 8;   // 256 B
static constexpr int kMaxShift = 30;  // 1 GiB

static unsigned long long blit_copy_kib(bool* is_default) {
  const char* env = std::getenv("GPU_FORCE_BLIT_COPY_SIZE");
  if (env == nullptr || env[0] == '\0') {
    *is_default = true;
    return kDefaultBlitKib;
  }
  *is_default = false;
  if (env[0] == '-') {
    std::cerr << "GPU_FORCE_BLIT_COPY_SIZE must be an unsigned KiB count, got '"
              << env << "'\n";
    std::exit(1);
  }
  errno = 0;
  char* end = nullptr;
  unsigned long long kib = std::strtoull(env, &end, 10);
  if (end == env || *end != '\0' || errno == ERANGE) {
    std::cerr << "GPU_FORCE_BLIT_COPY_SIZE must be an unsigned KiB count, got '"
              << env << "'\n";
    std::exit(1);
  }
  return kib;
}

static size_t blit_threshold_bytes(unsigned long long kib) {
  if (kib > std::numeric_limits<size_t>::max() / 1024) {
    std::cerr << "GPU_FORCE_BLIT_COPY_SIZE is too large to represent in bytes\n";
    std::exit(1);
  }
  return static_cast<size_t>(kib) * 1024;
}

static std::vector<size_t> copy_sizes() {
  std::vector<size_t> sizes;
  sizes.reserve(kMaxShift - kMinShift + 1);
  for (int shift = kMinShift; shift <= kMaxShift; ++shift) {
    sizes.push_back(size_t{1} << shift);
  }
  return sizes;
}

// Per-copy engine for same-device hipMalloc. Independent of sdmaCopyThreshold_.
static const char* copy_note(hipMemcpyKind kind) {
#if SDMA_HAS_NOCU
  if (kind == hipMemcpyDeviceToDeviceNoCU) return "SDMA";
#else
  (void)kind;
#endif
  return "shader blit";
}

// Host time for `repeat` ordered copies, after one untimed warmup copy.
static long long time_copy(uint8_t* dst, uint8_t* src, size_t bytes, int repeat,
                           hipMemcpyKind kind) {
  CHECK(hipMemcpyAsync(dst, src, bytes, kind, 0));
  CHECK(hipDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    CHECK(hipMemcpyAsync(dst, src, bytes, kind, 0));
  }
  CHECK(hipDeviceSynchronize());
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
  CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::cerr << "No HIP devices available\n";
    return 1;
  }
  CHECK(hipSetDevice(0));

  hipDeviceProp_t prop{};
  CHECK(hipGetDeviceProperties(&prop, 0));
  std::cout << "Device: " << prop.name
            << ", asyncEngineCount=" << prop.asyncEngineCount << std::endl;

  bool blit_is_default = false;
  const unsigned long long blit_kib = blit_copy_kib(&blit_is_default);
  const size_t blit_threshold = blit_threshold_bytes(blit_kib);
  const std::vector<size_t> sizes = copy_sizes();
  std::cout << "GPU_FORCE_BLIT_COPY_SIZE=" << blit_kib;
  if (blit_is_default) std::cout << " (unset, HIP default)";
  std::cout << ", sdmaCopyThreshold " << blit_threshold << " bytes." << std::endl;
  std::cout << "That threshold selects a shader blit for host-accessible copies "
               "with size <= threshold. It does not select the engine for these "
               "hipMalloc device-to-device copies." << std::endl;
  std::cout << "Sweep is powers of two from " << (size_t{1} << kMinShift)
            << " bytes to " << (size_t{1} << kMaxShift)
            << " bytes and does not depend on GPU_FORCE_BLIT_COPY_SIZE." << std::endl;

#if SDMA_HAS_NOCU
  std::cout << "hipMemcpyDeviceToDeviceNoCU requests SDMA at every size." << std::endl;
  const hipMemcpyKind kinds[] = {hipMemcpyDeviceToDeviceNoCU, hipMemcpyDeviceToDevice};
  const char* names[] = {"hipMemcpyDeviceToDeviceNoCU", "hipMemcpyDeviceToDevice"};
#else
  std::cout << "hipMemcpyDeviceToDeviceNoCU requires HIP 6.1 or newer; "
               "timing hipMemcpyDeviceToDevice only." << std::endl;
  const hipMemcpyKind kinds[] = {hipMemcpyDeviceToDevice};
  const char* names[] = {"hipMemcpyDeviceToDevice"};
#endif
  const int nkinds = static_cast<int>(sizeof(kinds) / sizeof(kinds[0]));

  bool ok = true;
  for (size_t bytes : sizes) {
    std::vector<uint8_t> h_src(bytes);
    std::vector<uint8_t> h_dst(bytes);
    fill_pattern(h_src.data(), bytes);

    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    CHECK(hipMalloc(reinterpret_cast<void**>(&d_src), bytes));
    CHECK(hipMalloc(reinterpret_cast<void**>(&d_dst), bytes));
    CHECK(hipMemcpy(d_src, h_src.data(), bytes, hipMemcpyHostToDevice));

    for (int k = 0; k < nkinds; ++k) {
      CHECK(hipMemset(d_dst, 0, bytes));
      long long ns = time_copy(d_dst, d_src, bytes, repeat, kinds[k]);
      report(names[k], copy_note(kinds[k]), bytes, ns, repeat);
      CHECK(hipMemcpy(h_dst.data(), d_dst, bytes, hipMemcpyDeviceToHost));
      ok = verify(h_dst.data(), bytes) && ok;
    }

    CHECK(hipFree(d_src));
    CHECK(hipFree(d_dst));
  }

  std::cout << (ok ? "PASS" : "FAIL") << std::endl;
  return ok ? 0 : 1;
}
