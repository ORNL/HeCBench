// One-sided put and get through a peer GPU pointer.
// The timed kernel is the flat load/store path from
// https://github.com/ROCm/mori/tree/main/benchmark/cco (transport "lsa"):
// a strided copy, a system fence, and a cross-block barrier on every iteration.
// Put stores into the peer. Get loads from the peer. The initiator is the
// first device that can access a peer.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sycl/sycl.hpp>

constexpr int kThreads = 256;
constexpr int kBandwidthBlocks = 32;
constexpr int kWarmup = 5;
constexpr size_t kMaxBytes = 32ull << 20;

// sycl::double4 is a class stored as a plain array, so its assignment becomes
// memcpy and NVPTX expands that to scalar 8-byte operations. A Clang vector
// is one 32-byte load and store, which NVPTX splits into 16-byte vector ops.
using double4_native = double __attribute__((ext_vector_type(4)));

using device_atomic = sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                                       sycl::memory_scope::device,
                                       sycl::access::address_space::global_space>;

template <typename Vec4>
static void copy_vec4(double* dst, const double* src, size_t n, int lane, int nlanes) {
  const size_t nvec = n / 4;
  Vec4* dst4 = reinterpret_cast<Vec4*>(dst);
  const Vec4* src4 = reinterpret_cast<const Vec4*>(src);
  for (size_t i = static_cast<size_t>(lane); i < nvec; i += static_cast<size_t>(nlanes))
    dst4[i] = src4[i];
  if (lane == 0) {
    for (size_t i = nvec * 4; i < n; ++i)
      dst[i] = src[i];
  }
}

static void copy_strided(double* dst, const double* src, size_t n, int lane, int nlanes) {
  const uintptr_t bits = reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src);
  if ((bits & (alignof(double4_native) - 1)) == 0)
    copy_vec4<double4_native>(dst, src, n, lane, nlanes);
  else {
    for (size_t i = static_cast<size_t>(lane); i < n; i += static_cast<size_t>(nlanes))
      dst[i] = src[i];
  }
}

static void grid_barrier(sycl::nd_item<1> item, unsigned int* counter, int nblocks, int round) {
  sycl::group_barrier(item.get_group());
  if (item.get_local_id(0) == 0) {
    device_atomic arrivals(counter[0]);
    device_atomic phase(counter[1]);
    const unsigned int c = arrivals.fetch_add(1u);
    if (c == static_cast<unsigned int>(nblocks * (round + 1) - 1))
      phase.fetch_add(1u);
    while (phase.load() != static_cast<unsigned int>(round + 1)) {
    }
  }
  sycl::group_barrier(item.get_group());
}

static void putget(sycl::nd_item<1> item, double* dst, const double* src, size_t n,
                   int iters, unsigned int* counter) {
  const int nblocks = item.get_group_range(0);
  const int bid = item.get_group(0);
  const int lane = item.get_local_id(0);
  const int nlanes = item.get_local_range(0);
  const size_t chunk = n / static_cast<size_t>(nblocks);
  const size_t begin = static_cast<size_t>(bid) * chunk;
  const size_t end = (bid == nblocks - 1) ? n : begin + chunk;

  for (int i = 0; i < iters; ++i) {
    copy_strided(dst + begin, src + begin, end - begin, lane, nlanes);
    // System fence forces each round's cross-GPU stores out, otherwise the
    // repeated same-region traffic is cache-absorbed and we'd time cache BW.
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    grid_barrier(item, counter, nblocks, i);
  }
}

static double pattern(size_t i) {
  return static_cast<double>(static_cast<uint32_t>(i * 1315423911u));
}

static bool matches(const double* got, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != pattern(i)) {
      fprintf(stderr, "mismatch at %zu: got %.0f expected %.0f\n",
              i, got[i], pattern(i));
      return false;
    }
  }
  return true;
}

static void launch_sync(sycl::queue& q, double* dst, const double* src, size_t n, int iters,
                   unsigned int* counter, int blocks) {
  q.memset(counter, 0, 2 * sizeof(unsigned int));
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(blocks * kThreads, kThreads),
                   [=](sycl::nd_item<1> item) {
                     putget(item, dst, src, n, iters, counter);
                   });
  }).wait();
}

static bool verify_once(sycl::queue& q, sycl::queue& reader, double* dst, const double* src,
                        size_t n, size_t bytes, unsigned int* counter, int blocks,
                        double* scratch) {
  launch_sync(q, dst, src, n, 1, counter, blocks);
  reader.memcpy(scratch, dst, bytes).wait();
  return matches(scratch, n);
}

static double time_kernel(sycl::queue& q, double* dst, const double* src, size_t n, int iters,
                      unsigned int* counter, int blocks) {
  q.memset(counter, 0, 2 * sizeof(unsigned int)).wait();
  const auto start = std::chrono::steady_clock::now();
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(blocks * kThreads, kThreads),
                   [=](sycl::nd_item<1> item) {
                     putget(item, dst, src, n, iters, counter);
                   });
  }).wait();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  if (repeat <= 0) {
    fprintf(stderr, "repeat must be positive\n");
    return 1;
  }

  auto devs = sycl::platform(sycl::gpu_selector_v).get_devices(sycl::info::device_type::gpu);
  const int gpu_n = static_cast<int>(devs.size());
  if (gpu_n < 2) {
    printf("Two GPUs with peer access are required. Waiving test.\n");
    return 0;
  }

  int initiator = -1;
  int peer = -1;
  for (int i = 0; i < gpu_n && initiator < 0; ++i) {
    for (int j = 0; j < gpu_n; ++j) {
      if (i == j) continue;
      bool can = devs[i].ext_oneapi_can_access_peer(
          devs[j], sycl::ext::oneapi::peer_access::access_supported);
      if (can) {
        initiator = i;
        peer = j;
        break;
      }
    }
  }
  if (initiator < 0) {
    printf("Peer access is not available. Waiving test.\n");
    return 0;
  }

  printf("One-sided put/get, initiator GPU%d (%s) peer GPU%d (%s), repeat %d\n",
         initiator, devs[initiator].get_info<sycl::info::device::name>().c_str(),
         peer, devs[peer].get_info<sycl::info::device::name>().c_str(), repeat);

  devs[initiator].ext_oneapi_enable_peer_access(devs[peer]);
  if (devs[peer].ext_oneapi_can_access_peer(
          devs[initiator], sycl::ext::oneapi::peer_access::access_supported)) {
    devs[peer].ext_oneapi_enable_peer_access(devs[initiator]);
  }

  sycl::queue q0(devs[initiator], sycl::property::queue::in_order());
  sycl::queue q1(devs[peer], sycl::property::queue::in_order());

  double* local = static_cast<double*>(sycl::malloc_device(kMaxBytes, q0));
  double* remote = static_cast<double*>(sycl::malloc_device(kMaxBytes, q1));
  unsigned int* counter = static_cast<unsigned int*>(sycl::malloc_device(2 * sizeof(unsigned int), q0));

  const size_t max_n = kMaxBytes / sizeof(double);
  std::vector<double> host(max_n);
  std::vector<double> scratch(max_n);
  bool ok = true;
  if (!local || !remote || !counter) {
    fprintf(stderr, "device allocation failed\n");
    ok = false;
    goto cleanup;
  }
  for (size_t i = 0; i < max_n; ++i) host[i] = pattern(i);

  printf("%12s %14s %14s %14s %14s\n",
         "bytes", "put_bw(GB/s)", "get_bw(GB/s)", "put_lat(us)", "get_lat(us)");

  for (size_t bytes = sizeof(double); bytes <= kMaxBytes; bytes *= 2) {
    const size_t n = bytes / sizeof(double);

    q0.memcpy(local, host.data(), bytes);
    q1.memset(remote, 0, bytes).wait();
    q0.wait();

    launch_sync(q0, remote, local, n, kWarmup, counter, kBandwidthBlocks);
    q1.memset(remote, 0, bytes).wait();
    if (!verify_once(q0, q1, remote, local, n, bytes, counter, kBandwidthBlocks, scratch.data())) {
      fprintf(stderr, "put verification failed at %zu bytes\n", bytes);
      ok = false;
      goto cleanup;
    }
    const double put_bw_ms = time_kernel(q0, remote, local, n, repeat, counter, kBandwidthBlocks);

    launch_sync(q0, remote, local, n, kWarmup, counter, 1);
    q1.memset(remote, 0, bytes).wait();
    if (!verify_once(q0, q1, remote, local, n, bytes, counter, 1, scratch.data())) {
      fprintf(stderr, "put latency verification failed at %zu bytes\n", bytes);
      ok = false;
      goto cleanup;
    }
    const double put_lat_ms = time_kernel(q0, remote, local, n, repeat, counter, 1);

    q1.memcpy(remote, host.data(), bytes).wait();
    q0.memset(local, 0, bytes).wait();

    launch_sync(q0, local, remote, n, kWarmup, counter, kBandwidthBlocks);
    q0.memset(local, 0, bytes).wait();
    if (!verify_once(q0, q0, local, remote, n, bytes, counter, kBandwidthBlocks, scratch.data())) {
      fprintf(stderr, "get verification failed at %zu bytes\n", bytes);
      ok = false;
      goto cleanup;
    }
    const double get_bw_ms = time_kernel(q0, local, remote, n, repeat, counter, kBandwidthBlocks);

    launch_sync(q0, local, remote, n, kWarmup, counter, 1);
    q0.memset(local, 0, bytes).wait();
    if (!verify_once(q0, q0, local, remote, n, bytes, counter, 1, scratch.data())) {
      fprintf(stderr, "get latency verification failed at %zu bytes\n", bytes);
      ok = false;
      goto cleanup;
    }
    const double get_lat_ms = time_kernel(q0, local, remote, n, repeat, counter, 1);

    const double put_bw = (static_cast<double>(bytes) * repeat) / (put_bw_ms * 1e6);
    const double get_bw = (static_cast<double>(bytes) * repeat) / (get_bw_ms * 1e6);
    const double put_lat = (put_lat_ms * 1e3) / repeat;
    const double get_lat = (get_lat_ms * 1e3) / repeat;
    printf("%12zu %14.2f %14.2f %14.3f %14.3f\n",
           bytes, put_bw, get_bw, put_lat, get_lat);
  }

cleanup:
  if (local) sycl::free(local, q0);
  if (counter) sycl::free(counter, q0);
  if (remote) sycl::free(remote, q1);

  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
