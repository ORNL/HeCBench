// Multi-GPU MoE combine. One MPI rank owns one GPU.
//
// Each rank maps its expert-output buffer into the others with HIP IPC and
// reads it from the combine kernel. That is the intra-node path measured by
// the MORI EP benchmark. MPI carries the IPC handle bytes and the host
// barriers. It does not move device buffers.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <mpi.h>
#include "../moe-combine-cuda/combine.cuh"

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err_ = (call);                                                  \
    if (err_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,        \
                   hipGetErrorString(err_));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Parse one decimal flag value, or exit if it is missing or out of range.
static int integer_arg(const char *text, const char *name, long minimum,
                       long maximum) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!text[0] || *end || value < minimum || value > maximum) {
    std::fprintf(stderr, "invalid %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

// Launch combine_kernel for a top-k of 1, 2, 4, or 8.
static void launch_combine(int topk, int blocks, int tokens, int hidden, const int *dest,
                           const int *slot, const std::uint16_t *const *stage, std::uint16_t *out) {
  switch (topk) {
  case 1:
    combine_kernel<1><<<blocks, 256>>>(tokens, hidden, dest, slot, stage, out);
    break;
  case 2:
    combine_kernel<2><<<blocks, 256>>>(tokens, hidden, dest, slot, stage, out);
    break;
  case 4:
    combine_kernel<4><<<blocks, 256>>>(tokens, hidden, dest, slot, stage, out);
    break;
  case 8:
    combine_kernel<8><<<blocks, 256>>>(tokens, hidden, dest, slot, stage, out);
    break;
  default:
    std::fprintf(stderr, "topk must be 1, 2, 4, or 8\n");
    std::exit(EXIT_FAILURE);
  }
  HIP_CHECK(hipGetLastError());
}

// Export a device allocation. A failure is returned with the per-thread error cleared.
static hipError_t ipc_get_handle(hipIpcMemHandle_t *handle, void *ptr) {
  const hipError_t err = hipIpcGetMemHandle(handle, ptr);
  if (err == hipSuccess)
    return hipSuccess;
  const hipError_t stored = hipGetLastError();
  return stored == hipSuccess ? err : stored;
}

// Import a peer allocation. A failure is returned with the per-thread error cleared.
static hipError_t ipc_open_handle(void **ptr, hipIpcMemHandle_t handle) {
  const hipError_t err =
      hipIpcOpenMemHandle(ptr, handle, hipIpcMemLazyEnablePeerAccess);
  if (err == hipSuccess)
    return hipSuccess;
  const hipError_t stored = hipGetLastError();
  return stored == hipSuccess ? err : stored;
}

// All-gather one byte blob per rank. Lengths are std::size_t and must fit in an MPI int count.
static std::vector<std::byte> allgather_bytes(const void *local, std::size_t local_bytes,
                                              const std::vector<std::size_t> &sizes,
                                              std::vector<int> *displs) {
  if (local_bytes > INT_MAX) {
    std::fprintf(stderr, "IPC handle is larger than MPI can send\n");
    std::exit(EXIT_FAILURE);
  }
  std::vector<int> counts(sizes.size());
  displs->assign(sizes.size(), 0);
  std::size_t total = 0;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] > INT_MAX || total > INT_MAX - sizes[i]) {
      std::fprintf(stderr, "IPC handle is larger than MPI can send\n");
      std::exit(EXIT_FAILURE);
    }
    counts[i] = sizes[i];
    (*displs)[i] = total;
    total += sizes[i];
  }
  std::vector<std::byte> all(total > 0 ? total : 1);
  std::byte dummy{};
  const void *send = local_bytes == 0 ? &dummy : local;
  MPI_Allgatherv(send, local_bytes, MPI_BYTE, all.data(), counts.data(), displs->data(),
                 MPI_BYTE, MPI_COMM_WORLD);
  if (total == 0)
    all.clear();
  return all;
}

// Map peer expert buffers, time the combine, and check it against the host reference.
int main(int argc, char **argv) {
  int tokens = 4096;
  int hidden = 7168;
  int topk = 8;
  int experts = 32;
  int iterations = 100;
  int warmup = 100;
  for (int i = 1; i < argc; ++i) {
    const bool has_value = i + 1 < argc;
    if (has_value && !std::strcmp(argv[i], "--tokens"))
      tokens = integer_arg(argv[++i], "token count", 1, 1 << 20);
    else if (has_value && !std::strcmp(argv[i], "--hidden"))
      hidden = integer_arg(argv[++i], "hidden size", 8, 16384);
    else if (has_value && !std::strcmp(argv[i], "--topk"))
      topk = integer_arg(argv[++i], "top-k", 1, 8);
    else if (has_value && !std::strcmp(argv[i], "--experts"))
      experts = integer_arg(argv[++i], "experts per rank", 1, 4096);
    else if (has_value && !std::strcmp(argv[i], "--iters"))
      iterations = integer_arg(argv[++i], "iteration count", 1, 100000);
    else if (has_value && !std::strcmp(argv[i], "--warmup"))
      warmup = integer_arg(argv[++i], "warmup count", 0, 100000);
    else {
      std::fprintf(stderr,
                   "usage: %s [--tokens N] [--hidden N] [--topk 1|2|4|8] "
                   "[--experts N] [--iters N] [--warmup N]\n",
                   argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (hidden % 8 != 0) {
    std::fprintf(stderr, "hidden size must be a multiple of 8\n");
    return EXIT_FAILURE;
  }
  if (topk != 1 && topk != 2 && topk != 4 && topk != 8) {
    std::fprintf(stderr, "topk must be 1, 2, 4, or 8\n");
    return EXIT_FAILURE;
  }

  MPI_Init(&argc, &argv);
  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  if (world > 256) {
    if (rank == 0)
      std::fprintf(stderr, "at most 256 MPI ranks are supported\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  int devices = 0;
  HIP_CHECK(hipGetDeviceCount(&devices));
  if (devices < 1) {
    if (rank == 0)
      std::fprintf(stderr, "no GPU was found\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  HIP_CHECK(hipSetDevice(rank % devices));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, rank % devices));

  const CombineRoute route = build_route(world, tokens, topk, hidden, experts);
  const int slots = route.recv[rank];
  const std::size_t stage_bytes =
      static_cast<std::size_t>(slots) * hidden * sizeof(std::uint16_t);
  const std::size_t stage_alloc = std::max(stage_bytes, sizeof(std::uint16_t));
  const std::size_t out_bytes =
      static_cast<std::size_t>(tokens) * hidden * sizeof(std::uint16_t);
  const std::size_t map_n = static_cast<std::size_t>(tokens) * topk;

  std::uint16_t *stage = nullptr;
  std::uint16_t *out = nullptr;
  unsigned long long *flags = nullptr;
  int *dest = nullptr;
  int *slot = nullptr;
  HIP_CHECK(hipMalloc(&stage, stage_alloc));
  HIP_CHECK(hipMalloc(&out, out_bytes));
  // A failed fine-grained allocation stays in the per-thread last-error slot.
  // Clear it before the ordinary fallback, or a later hipGetLastError reports it.
  if (hipExtMallocWithFlags(reinterpret_cast<void **>(&flags),
                            sizeof(unsigned long long) * world,
                            hipDeviceMallocFinegrained) != hipSuccess) {
    (void)hipGetLastError();
    HIP_CHECK(hipMalloc(&flags, sizeof(unsigned long long) * world));
  }
  HIP_CHECK(hipMalloc(&dest, sizeof(int) * map_n));
  HIP_CHECK(hipMalloc(&slot, sizeof(int) * map_n));
  HIP_CHECK(hipMemset(flags, 0, sizeof(unsigned long long) * world));
  const std::size_t base = route.index(rank, 0, 0);
  HIP_CHECK(hipMemcpy(dest, route.dest.data() + base, sizeof(int) * map_n,
                    hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(slot, route.slot.data() + base, sizeof(int) * map_n,
                    hipMemcpyHostToDevice));

  std::vector<std::uint16_t *> stage_ptrs(world, nullptr);
  std::vector<unsigned long long *> flag_ptrs(world, nullptr);
  std::vector<void *> opened;
  std::uint16_t **d_stage_ptrs = nullptr;
  unsigned long long **d_flag_ptrs = nullptr;
  // Close imports, then free this rank's allocations.
  auto release_device = [&] {
    if (d_stage_ptrs)
      HIP_CHECK(hipFree(d_stage_ptrs));
    if (d_flag_ptrs)
      HIP_CHECK(hipFree(d_flag_ptrs));
    for (void *p : opened)
      (void)hipIpcCloseMemHandle(p);
    opened.clear();
    // Every rank drops its imports before any rank frees the exported buffers.
    MPI_Barrier(MPI_COMM_WORLD);
    HIP_CHECK(hipFree(stage));
    HIP_CHECK(hipFree(out));
    HIP_CHECK(hipFree(flags));
    HIP_CHECK(hipFree(dest));
    HIP_CHECK(hipFree(slot));
  };
  if (world > 1) {
    hipIpcMemHandle_t stage_handle{};
    hipIpcMemHandle_t flag_handle{};
    int local_ok = ipc_get_handle(&stage_handle, stage) == hipSuccess &&
                   ipc_get_handle(&flag_handle, flags) == hipSuccess;
    int all_exported = 0;
    MPI_Allreduce(&local_ok, &all_exported, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    int opened_ok = 0;
    if (all_exported) {
      const std::vector<std::size_t> handle_sizes(world, sizeof(hipIpcMemHandle_t));
      std::vector<int> stage_displ;
      std::vector<int> flag_displ;
      const std::vector<std::byte> stage_raw =
          allgather_bytes(&stage_handle, sizeof(stage_handle), handle_sizes, &stage_displ);
      const std::vector<std::byte> flag_raw =
          allgather_bytes(&flag_handle, sizeof(flag_handle), handle_sizes, &flag_displ);
      std::vector<hipIpcMemHandle_t> stage_handles(static_cast<std::size_t>(world));
      std::vector<hipIpcMemHandle_t> flag_handles(static_cast<std::size_t>(world));
      for (int p = 0; p < world; ++p) {
        std::memcpy(&stage_handles[static_cast<std::size_t>(p)],
                    stage_raw.data() + stage_displ[static_cast<std::size_t>(p)],
                    sizeof(hipIpcMemHandle_t));
        std::memcpy(&flag_handles[static_cast<std::size_t>(p)],
                    flag_raw.data() + flag_displ[static_cast<std::size_t>(p)],
                    sizeof(hipIpcMemHandle_t));
      }
      stage_ptrs[rank] = stage;
      flag_ptrs[rank] = flags;
      bool open_ok = true;
      for (int p = 0; p < world && open_ok; ++p) {
        if (p == rank)
          continue;
        void *sp = nullptr;
        void *fp = nullptr;
        if (ipc_open_handle(&sp, stage_handles[p]) != hipSuccess ||
            ipc_open_handle(&fp, flag_handles[p]) != hipSuccess) {
          if (sp)
            (void)hipIpcCloseMemHandle(sp);
          if (fp)
            (void)hipIpcCloseMemHandle(fp);
          open_ok = false;
          break;
        }
        stage_ptrs[p] = static_cast<std::uint16_t *>(sp);
        flag_ptrs[p] = static_cast<unsigned long long *>(fp);
        opened.push_back(sp);
        opened.push_back(fp);
      }
      opened_ok = open_ok ? 1 : 0;
    }
    int opened_all = 0;
    MPI_Allreduce(&opened_ok, &opened_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!opened_all) {
      for (void *p : opened)
        (void)hipIpcCloseMemHandle(p);
      opened.clear();
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0)
        std::fprintf(stderr, "peer transport is not available between these GPUs\n");
      release_device();
      MPI_Finalize();
      return EXIT_FAILURE;
    }
  } else {
    stage_ptrs[rank] = stage;
    flag_ptrs[rank] = flags;
  }

  HIP_CHECK(hipMalloc(&d_stage_ptrs, sizeof(std::uint16_t *) * world));
  HIP_CHECK(hipMalloc(&d_flag_ptrs, sizeof(unsigned long long *) * world));
  HIP_CHECK(hipMemcpy(d_stage_ptrs, stage_ptrs.data(), sizeof(std::uint16_t *) * world,
                    hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_flag_ptrs, flag_ptrs.data(), sizeof(unsigned long long *) * world,
                    hipMemcpyHostToDevice));

  const int fill_blocks = 256;
  fill_stage<<<fill_blocks, 256>>>(stage, rank, slots, hidden);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  const int blocks = tokens < 256 ? tokens : 256;
  unsigned long long epoch = 1;
  // One pair of kernels at a time. The rendezvous is its own one-block launch
  // so the combine grid never spins on a block the device has not scheduled.
  // The wait accepts a later epoch: the next iteration on a faster rank can
  // overwrite the flag before a slower rank observes this one.
  auto step = [&] {
    if (world > 1)
      peer_rendezvous<<<1, 256>>>(flags, d_flag_ptrs, rank, world, epoch);
    launch_combine(topk, blocks, tokens, hidden, dest, slot, d_stage_ptrs, out);
    ++epoch;
    HIP_CHECK(hipDeviceSynchronize());
  };
  for (int i = 0; i < warmup; ++i)
    step();
  MPI_Barrier(MPI_COMM_WORLD);

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i)
    step();
  const auto t1 = std::chrono::steady_clock::now();

  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
  std::vector<std::uint16_t> host(static_cast<std::size_t>(tokens) * hidden);
  HIP_CHECK(hipMemcpy(host.data(), out, out_bytes, hipMemcpyDeviceToHost));
  const double local_error = max_abs_error(route, rank, host.data(), 8);
  const bool local_pass = std::isfinite(local_error) && local_error < 2.0e-2;

  const double local_algo = algo_bytes_of(route, rank);
  const double local_fabric = fabric_bytes_of(route, rank);
  const int local_recv = route.recv[rank];
  std::vector<double> all_ms(static_cast<std::size_t>(world));
  std::vector<double> all_algo(static_cast<std::size_t>(world));
  std::vector<double> all_fabric(static_cast<std::size_t>(world));
  std::vector<int> all_recv(static_cast<std::size_t>(world));
  double max_error = 0;
  int all_pass = 0;
  const int pass_bit = local_pass ? 1 : 0;
  MPI_Gather(&ms, 1, MPI_DOUBLE, rank == 0 ? all_ms.data() : nullptr, 1, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);
  MPI_Gather(&local_algo, 1, MPI_DOUBLE, rank == 0 ? all_algo.data() : nullptr, 1, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);
  MPI_Gather(&local_fabric, 1, MPI_DOUBLE, rank == 0 ? all_fabric.data() : nullptr, 1,
             MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&local_recv, 1, MPI_INT, rank == 0 ? all_recv.data() : nullptr, 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Allreduce(&pass_bit, &all_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (rank == 0) {
    const int slow = slowest_rank(all_ms.data(), all_fabric.data(), all_algo.data(), world);
    const double max_ms = all_ms[static_cast<std::size_t>(slow)];
    const double seconds = max_ms / 1.0e3;
    std::printf("device: %s\n", prop.name);
    std::printf("ranks: %d  tokens: %d  hidden: %d  topk: %d  experts_per_rank: %d\n",
                world, tokens, hidden, topk, experts);
    std::printf("transport: peer\n");
    std::printf("recv_tokens: %d  iterations: %d  warmup: %d\n",
                all_recv[static_cast<std::size_t>(slow)], iterations, warmup);
    std::printf("kernel: %.6f ms\n", max_ms);
    std::printf("algo_bandwidth: %.3f GB/s\n",
                all_algo[static_cast<std::size_t>(slow)] / 1.0e9 / seconds);
    std::printf("fabric_bandwidth: %.3f GB/s\n",
                all_fabric[static_cast<std::size_t>(slow)] / 1.0e9 / seconds);
    std::printf("max_abs_error: %.3e\n", max_error);
    std::printf("moe-combine: %s\n", all_pass ? "PASS" : "FAIL");
  }

  release_device();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
