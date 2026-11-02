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
#include <cuda_runtime.h>

#define GPU_CHECK(x) do {                                                    \
  cudaError_t err_ = (x);                                                    \
  if (err_ != cudaSuccess) {                                                 \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
            cudaGetErrorString(err_));                                       \
    exit(1);                                                                 \
  }                                                                          \
} while (0)

constexpr int kThreads = 256;
constexpr int kBandwidthBlocks = 32;
constexpr int kWarmup = 5;
constexpr size_t kMaxBytes = 32ull << 20;

template <typename Vec4>
__device__ void copy_vec4(double* __restrict__ dst, const double* __restrict__ src,
                          size_t n, int lane, int nlanes) {
  const size_t nvec = n / 4;
  Vec4* __restrict__ dst4 = reinterpret_cast<Vec4*>(dst);
  const Vec4* __restrict__ src4 = reinterpret_cast<const Vec4*>(src);
  for (size_t i = static_cast<size_t>(lane); i < nvec; i += static_cast<size_t>(nlanes))
    dst4[i] = src4[i];
  if (lane == 0) {
    for (size_t i = nvec * 4; i < n; ++i)
      dst[i] = src[i];
  }
}

__device__ void copy_strided(double* __restrict__ dst, const double* __restrict__ src,
                             size_t n, int lane, int nlanes) {
  const uintptr_t bits = reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src);
#if CUDART_VERSION >= 13000
  // double4_32a requires 32-byte alignment. double4_16a is the 16-byte-aligned
  // 32-byte vector. Otherwise copy one double at a time.
  if ((bits & (alignof(double4_32a) - 1)) == 0)
    copy_vec4<double4_32a>(dst, src, n, lane, nlanes);
  else if ((bits & (alignof(double4_16a) - 1)) == 0)
    copy_vec4<double4_16a>(dst, src, n, lane, nlanes);
  else
#else
  if ((bits & (alignof(double4) - 1)) == 0)
    copy_vec4<double4>(dst, src, n, lane, nlanes);
  else
#endif
  {
    for (size_t i = static_cast<size_t>(lane); i < n; i += static_cast<size_t>(nlanes))
      dst[i] = src[i];
  }
}

// Arrival counter in counter[0], phase in counter[1]. Both start at 0 and
// accumulate across iterations of one launch. The phase wait is an atomic
// load, matching the SYCL barrier, so it observes the atomic update.
__device__ void grid_barrier(unsigned int* counter, int nblocks, int round) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int c = atomicAdd(counter, 1u);
    if (c == static_cast<unsigned int>(nblocks * (round + 1) - 1))
      atomicAdd(counter + 1, 1u);
    while (atomicAdd(counter + 1, 0u) != static_cast<unsigned int>(round + 1)) {
    }
  }
  __syncthreads();
}

__global__ void putget(double* __restrict__ dst, const double* __restrict__ src,
                       size_t n, int iters, unsigned int* counter) {
  const int nblocks = gridDim.x;
  const int bid = blockIdx.x;
  const size_t chunk = n / static_cast<size_t>(nblocks);
  const size_t begin = static_cast<size_t>(bid) * chunk;
  const size_t end = (bid == nblocks - 1) ? n : begin + chunk;

  for (int i = 0; i < iters; ++i) {
    copy_strided(dst + begin, src + begin, end - begin, threadIdx.x, blockDim.x);
    // System fence forces each round's cross-GPU stores out, otherwise the
    // repeated same-region traffic is cache-absorbed and we'd time cache BW.
    __threadfence_system();
    grid_barrier(counter, nblocks, i);
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

static void launch_sync(double* dst, const double* src, size_t n, int iters,
                        unsigned int* counter, int blocks) {
  GPU_CHECK(cudaMemset(counter, 0, 2 * sizeof(unsigned int)));
  putget<<<blocks, kThreads>>>(dst, src, n, iters, counter);
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaDeviceSynchronize());
}

static bool verify_once(double* dst, const double* src, size_t n, size_t bytes,
                        unsigned int* counter, int blocks, double* scratch) {
  launch_sync(dst, src, n, 1, counter, blocks);
  GPU_CHECK(cudaMemcpy(scratch, dst, bytes, cudaMemcpyDeviceToHost));
  return matches(scratch, n);
}

static double time_kernel(double* dst, const double* src, size_t n, int iters,
                          unsigned int* counter, int blocks) {
  GPU_CHECK(cudaMemset(counter, 0, 2 * sizeof(unsigned int)));
  GPU_CHECK(cudaDeviceSynchronize());
  const auto start = std::chrono::steady_clock::now();
  putget<<<blocks, kThreads>>>(dst, src, n, iters, counter);
  GPU_CHECK(cudaDeviceSynchronize());
  const auto stop = std::chrono::steady_clock::now();
  GPU_CHECK(cudaGetLastError());
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

  int gpu_n = 0;
  GPU_CHECK(cudaGetDeviceCount(&gpu_n));
  if (gpu_n < 2) {
    printf("Two GPUs with peer access are required. Waiving test.\n");
    return 0;
  }

  int initiator = -1;
  int peer = -1;
  for (int i = 0; i < gpu_n && initiator < 0; ++i) {
    for (int j = 0; j < gpu_n; ++j) {
      if (i == j) continue;
      int can = 0;
      GPU_CHECK(cudaDeviceCanAccessPeer(&can, i, j));
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

  cudaDeviceProp prop_i, prop_p;
  GPU_CHECK(cudaGetDeviceProperties(&prop_i, initiator));
  GPU_CHECK(cudaGetDeviceProperties(&prop_p, peer));
  printf("One-sided put/get, initiator GPU%d (%s) peer GPU%d (%s), repeat %d\n",
         initiator, prop_i.name, peer, prop_p.name, repeat);

  GPU_CHECK(cudaSetDevice(initiator));
  GPU_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
  GPU_CHECK(cudaSetDevice(peer));
  int reverse = 0;
  GPU_CHECK(cudaDeviceCanAccessPeer(&reverse, peer, initiator));
  if (reverse) GPU_CHECK(cudaDeviceEnablePeerAccess(initiator, 0));
  GPU_CHECK(cudaSetDevice(initiator));

  double* local = nullptr;
  double* remote = nullptr;
  unsigned int* counter = nullptr;
  GPU_CHECK(cudaMalloc(&local, kMaxBytes));
  GPU_CHECK(cudaMalloc(&counter, 2 * sizeof(unsigned int)));
  GPU_CHECK(cudaSetDevice(peer));
  GPU_CHECK(cudaMalloc(&remote, kMaxBytes));
  GPU_CHECK(cudaSetDevice(initiator));

  const size_t max_n = kMaxBytes / sizeof(double);
  std::vector<double> host(max_n);
  std::vector<double> scratch(max_n);
  for (size_t i = 0; i < max_n; ++i) host[i] = pattern(i);

  printf("%12s %14s %14s %14s %14s\n",
         "bytes", "put_bw(GB/s)", "get_bw(GB/s)", "put_lat(us)", "get_lat(us)");

  bool ok = true;
  for (size_t bytes = sizeof(double); bytes <= kMaxBytes; bytes *= 2) {
    const size_t n = bytes / sizeof(double);

    GPU_CHECK(cudaMemcpy(local, host.data(), bytes, cudaMemcpyHostToDevice));
    GPU_CHECK(cudaSetDevice(peer));
    GPU_CHECK(cudaMemset(remote, 0, bytes));
    GPU_CHECK(cudaDeviceSynchronize());
    GPU_CHECK(cudaSetDevice(initiator));

    launch_sync(remote, local, n, kWarmup, counter, kBandwidthBlocks);
    GPU_CHECK(cudaSetDevice(peer));
    GPU_CHECK(cudaMemset(remote, 0, bytes));
    GPU_CHECK(cudaDeviceSynchronize());
    GPU_CHECK(cudaSetDevice(initiator));
    if (!verify_once(remote, local, n, bytes, counter, kBandwidthBlocks, scratch.data())) {
      fprintf(stderr, "put verification failed at %zu bytes\n", bytes);
      ok = false;
      goto cleanup;
    }
    const double put_bw_ms = time_kernel(remote, local, n, repeat, counter, kBandwidthBlocks);

    launch_sync(remote, local, n, kWarmup, counter, 1);
    GPU_CHECK(cudaSetDevice(peer));
    GPU_CHECK(cudaMemset(remote, 0, bytes));
    GPU_CHECK(cudaDeviceSynchronize());
    GPU_CHECK(cudaSetDevice(initiator));
    if (!verify_once(remote, local, n, bytes, counter, 1, scratch.data())) {
      fprintf(stderr, "put latency verification failed at %zu bytes\n", bytes);
      ok = false;
      goto cleanup;
    }
    const double put_lat_ms = time_kernel(remote, local, n, repeat, counter, 1);

    GPU_CHECK(cudaSetDevice(peer));
    GPU_CHECK(cudaMemcpy(remote, host.data(), bytes, cudaMemcpyHostToDevice));
    GPU_CHECK(cudaDeviceSynchronize());
    GPU_CHECK(cudaSetDevice(initiator));
    GPU_CHECK(cudaMemset(local, 0, bytes));

    launch_sync(local, remote, n, kWarmup, counter, kBandwidthBlocks);
    GPU_CHECK(cudaMemset(local, 0, bytes));
    if (!verify_once(local, remote, n, bytes, counter, kBandwidthBlocks, scratch.data())) {
      fprintf(stderr, "get verification failed at %zu bytes\n", bytes);
      ok = false;
      goto cleanup;
    }
    const double get_bw_ms = time_kernel(local, remote, n, repeat, counter, kBandwidthBlocks);

    launch_sync(local, remote, n, kWarmup, counter, 1);
    GPU_CHECK(cudaMemset(local, 0, bytes));
    if (!verify_once(local, remote, n, bytes, counter, 1, scratch.data())) {
      fprintf(stderr, "get latency verification failed at %zu bytes\n", bytes);
      ok = false;
      goto cleanup;
    }
    const double get_lat_ms = time_kernel(local, remote, n, repeat, counter, 1);

    const double put_bw = (static_cast<double>(bytes) * repeat) / (put_bw_ms * 1e6);
    const double get_bw = (static_cast<double>(bytes) * repeat) / (get_bw_ms * 1e6);
    const double put_lat = (put_lat_ms * 1e3) / repeat;
    const double get_lat = (get_lat_ms * 1e3) / repeat;
    printf("%12zu %14.2f %14.2f %14.3f %14.3f\n",
           bytes, put_bw, get_bw, put_lat, get_lat);
  }

cleanup:
  GPU_CHECK(cudaFree(local));
  GPU_CHECK(cudaFree(counter));
  GPU_CHECK(cudaSetDevice(peer));
  GPU_CHECK(cudaFree(remote));

  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
