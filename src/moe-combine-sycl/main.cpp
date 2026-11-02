// Multi-GPU MoE combine. One MPI rank owns one GPU.
//
// Each rank maps its expert-output buffer into the others with
// sycl::ext::oneapi::experimental::ipc::memory and reads it from the combine
// kernel. That is the intra-node path measured by the MORI EP benchmark. MPI
// carries the IPC handle bytes and the host barriers. It does not move
// device buffers.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include "combine.hpp"
#include "reference.hpp"

namespace ipc_mem = sycl::ext::oneapi::experimental::ipc::memory;
using ipc_handle = sycl::ext::oneapi::experimental::ipc::handle;
using ipc_bytes = sycl::ext::oneapi::experimental::ipc::handle_data_t;

static_assert(sizeof(std::size_t) == 8, "handle lengths are exchanged with MPI_UINT64_T");

// Parse one decimal flag value, or exit if it is missing or out of range.
static int integer_arg(const char *text, const char *name, long minimum, long maximum) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!text[0] || *end || value < minimum || value > maximum) {
    std::fprintf(stderr, "invalid %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
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

// Enqueue combine_kernel for one top-k.
template <int kTopk>
static void launch_combine(sycl::queue &q, int blocks, int tokens, int hidden, const int *dest,
                           const int *slot, const std::uint16_t *const *stage, std::uint16_t *out) {
  q.parallel_for(sycl::nd_range<1>(static_cast<std::size_t>(blocks) * 256, 256),
                 [=](sycl::nd_item<1> item) {
                   combine_kernel<kTopk>(item, tokens, hidden, dest, slot, stage, out);
                 });
}

// Enqueue combine_kernel for a top-k of 1, 2, 4, or 8.
static void launch_combine(sycl::queue &q, int topk, int blocks, int tokens, int hidden,
                           const int *dest, const int *slot, const std::uint16_t *const *stage,
                           std::uint16_t *out) {
  switch (topk) {
  case 1:
    launch_combine<1>(q, blocks, tokens, hidden, dest, slot, stage, out);
    break;
  case 2:
    launch_combine<2>(q, blocks, tokens, hidden, dest, slot, stage, out);
    break;
  case 4:
    launch_combine<4>(q, blocks, tokens, hidden, dest, slot, stage, out);
    break;
  case 8:
    launch_combine<8>(q, blocks, tokens, hidden, dest, slot, stage, out);
    break;
  default:
    std::fprintf(stderr, "topk must be 1, 2, 4, or 8\n");
    std::exit(EXIT_FAILURE);
  }
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
  if (hidden % 8 != 0 || (topk != 1 && topk != 2 && topk != 4 && topk != 8)) {
    std::fprintf(stderr, "hidden size must be a multiple of 8 and topk must be 1, 2, 4, or 8\n");
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

  try {
    const auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
      if (rank == 0)
        std::fprintf(stderr, "no GPU was found\n");
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    const sycl::device dev = gpus[rank % static_cast<int>(gpus.size())];
    sycl::queue q{dev, sycl::property::queue::in_order()};
    const sycl::context ctx = q.get_context();
    const std::string name = dev.get_info<sycl::info::device::name>();

    const CombineRoute route = build_route(world, tokens, topk, hidden, experts);
    const int slots = route.recv[rank];
    const std::size_t stage_n =
        std::max(static_cast<std::size_t>(slots) * hidden, std::size_t{1});
    const std::size_t out_n = static_cast<std::size_t>(tokens) * hidden;
    const std::size_t map_n = static_cast<std::size_t>(tokens) * topk;

    std::uint16_t *stage = sycl::malloc_device<std::uint16_t>(stage_n, q);
    std::uint16_t *out = sycl::malloc_device<std::uint16_t>(out_n, q);
    unsigned long long *flags = sycl::malloc_device<unsigned long long>(world, q);
    int *dest = sycl::malloc_device<int>(map_n, q);
    int *slot = sycl::malloc_device<int>(map_n, q);
    if (!stage || !out || !flags || !dest || !slot)
      throw std::bad_alloc();
    q.memset(flags, 0, sizeof(unsigned long long) * world);
    const std::size_t base = route.index(rank, 0, 0);
    q.memcpy(dest, route.dest.data() + base, sizeof(int) * map_n);
    q.memcpy(slot, route.slot.data() + base, sizeof(int) * map_n);

    std::vector<std::uint16_t *> stage_ptrs(world, nullptr);
    std::vector<unsigned long long *> flag_ptrs(world, nullptr);
    std::vector<void *> opened;
    std::vector<ipc_handle> exports;
    exports.reserve(2);

    // Drop exported handle references after every rank has opened them.
    auto put_exports = [&] {
      for (ipc_handle &handle : exports)
        ipc_mem::put(handle, ctx);
      exports.clear();
    };
    // Close peer allocations imported by this rank.
    auto close_opened = [&] {
      for (void *p : opened)
        ipc_mem::close(p, ctx);
      opened.clear();
    };
    std::uint16_t **d_stage_ptrs = nullptr;
    unsigned long long **d_flag_ptrs = nullptr;
    // Close imports, then free this rank's allocations.
    auto release_device = [&] {
      close_opened();
      if (d_stage_ptrs)
        sycl::free(d_stage_ptrs, q);
      if (d_flag_ptrs)
        sycl::free(d_flag_ptrs, q);
      // Every rank drops its imports before any rank frees the exported buffers.
      MPI_Barrier(MPI_COMM_WORLD);
      sycl::free(stage, q);
      sycl::free(out, q);
      sycl::free(flags, q);
      sycl::free(dest, q);
      sycl::free(slot, q);
    };

    if (world > 1) {
      ipc_bytes stage_bytes;
      ipc_bytes flag_bytes;
      int local_ok = 0;
      try {
        if (dev.has(sycl::aspect::ext_oneapi_ipc_memory)) {
          exports.push_back(ipc_mem::get(stage, ctx));
          exports.push_back(ipc_mem::get(flags, ctx));
          stage_bytes = exports[0].data();
          flag_bytes = exports[1].data();
          local_ok = !stage_bytes.empty() && !flag_bytes.empty();
        }
      } catch (const sycl::exception &) {
        local_ok = 0;
      }
      if (!local_ok) {
        stage_bytes.clear();
        flag_bytes.clear();
      }
      std::vector<int> oks(world, 0);
      MPI_Allgather(&local_ok, 1, MPI_INT, oks.data(), 1, MPI_INT, MPI_COMM_WORLD);
      bool all_exported = true;
      for (int ok : oks)
        all_exported = all_exported && ok;

      int opened_ok = 0;
      if (all_exported) {
        std::size_t local_sizes[2] = {stage_bytes.size(), flag_bytes.size()};
        std::vector<std::size_t> all_sizes(static_cast<std::size_t>(world) * 2);
        MPI_Allgather(local_sizes, 2, MPI_UINT64_T, all_sizes.data(), 2, MPI_UINT64_T,
                      MPI_COMM_WORLD);
        std::vector<std::size_t> stage_sizes(world), flag_sizes(world);
        for (int p = 0; p < world; ++p) {
          stage_sizes[p] = all_sizes[static_cast<std::size_t>(p) * 2];
          flag_sizes[p] = all_sizes[static_cast<std::size_t>(p) * 2 + 1];
        }
        std::vector<int> stage_displ;
        std::vector<int> flag_displ;
        const std::vector<std::byte> all_stage = allgather_bytes(
            stage_bytes.empty() ? nullptr : stage_bytes.data(), stage_bytes.size(), stage_sizes,
            &stage_displ);
        const std::vector<std::byte> all_flags = allgather_bytes(
            flag_bytes.empty() ? nullptr : flag_bytes.data(), flag_bytes.size(), flag_sizes,
            &flag_displ);
        stage_ptrs[rank] = stage;
        flag_ptrs[rank] = flags;
        bool all_ok = true;
        for (int p = 0; p < world && all_ok; ++p) {
          if (p == rank)
            continue;
          void *sp = nullptr;
          void *fp = nullptr;
          try {
            const ipc_bytes stage_h(all_stage.begin() + stage_displ[p],
                                    all_stage.begin() + stage_displ[p] + stage_sizes[p]);
            const ipc_bytes flag_h(all_flags.begin() + flag_displ[p],
                                   all_flags.begin() + flag_displ[p] + flag_sizes[p]);
            sp = ipc_mem::open(stage_h, ctx, dev);
            fp = ipc_mem::open(flag_h, ctx, dev);
            stage_ptrs[p] = static_cast<std::uint16_t *>(sp);
            flag_ptrs[p] = static_cast<unsigned long long *>(fp);
            opened.push_back(sp);
            opened.push_back(fp);
          } catch (const sycl::exception &) {
            if (sp)
              ipc_mem::close(sp, ctx);
            all_ok = false;
          }
        }
        opened_ok = all_ok ? 1 : 0;
      }
      int opened_all = 0;
      MPI_Allreduce(&opened_ok, &opened_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      // Opened peer pointers stay valid after put. The exporting allocation
      // has to stay allocated until those peers close.
      put_exports();
      if (!opened_all) {
        close_opened();
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

    d_stage_ptrs = sycl::malloc_device<std::uint16_t *>(world, q);
    d_flag_ptrs = sycl::malloc_device<unsigned long long *>(world, q);
    if (!d_stage_ptrs || !d_flag_ptrs)
      throw std::bad_alloc();
    q.memcpy(d_stage_ptrs, stage_ptrs.data(), sizeof(std::uint16_t *) * world);
    q.memcpy(d_flag_ptrs, flag_ptrs.data(), sizeof(unsigned long long *) * world);
    const int fill_blocks = 256;
    q.parallel_for(sycl::nd_range<1>(static_cast<std::size_t>(fill_blocks) * 256, 256),
                   [=](sycl::nd_item<1> item) { fill_stage(item, stage, rank, slots, hidden); })
        .wait();
    MPI_Barrier(MPI_COMM_WORLD);

    const int blocks = tokens < 256 ? tokens : 256;
    unsigned long long epoch = 1;
    // One pair of kernels at a time. The rendezvous is its own one-group launch
    // so the combine grid never spins on a group the device has not scheduled.
    // The wait accepts a later epoch: the next iteration on a faster rank can
    // overwrite the flag before a slower rank observes this one.
    auto step = [&] {
      if (world > 1) {
        q.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> item) {
          peer_rendezvous(item, flags, d_flag_ptrs, rank, world, epoch);
        });
      }
      launch_combine(q, topk, blocks, tokens, hidden, dest, slot, d_stage_ptrs, out);
      ++epoch;
      q.wait_and_throw();
    };
    for (int i = 0; i < warmup; ++i)
      step();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
      step();
    const auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
    std::vector<std::uint16_t> host(out_n);
    q.memcpy(host.data(), out, out_n * sizeof(std::uint16_t)).wait();
    const double local_error = max_abs_error(route, rank, host.data(), 8);
    const int pass_bit = std::isfinite(local_error) && local_error < 2.0e-2;
    const double local_algo = algo_bytes_of(route, rank);
    const double local_fabric = fabric_bytes_of(route, rank);
    const int local_recv = route.recv[rank];
    std::vector<double> all_ms(static_cast<std::size_t>(world));
    std::vector<double> all_algo(static_cast<std::size_t>(world));
    std::vector<double> all_fabric(static_cast<std::size_t>(world));
    std::vector<int> all_recv(static_cast<std::size_t>(world));
    double max_error = 0;
    int all_pass = 0;
    MPI_Gather(&ms, 1, MPI_DOUBLE, rank == 0 ? all_ms.data() : nullptr, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    MPI_Gather(&local_algo, 1, MPI_DOUBLE, rank == 0 ? all_algo.data() : nullptr, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    MPI_Gather(&local_fabric, 1, MPI_DOUBLE, rank == 0 ? all_fabric.data() : nullptr, 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    MPI_Gather(&local_recv, 1, MPI_INT, rank == 0 ? all_recv.data() : nullptr, 1, MPI_INT, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&pass_bit, &all_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
      const int slow = slowest_rank(all_ms.data(), all_fabric.data(), all_algo.data(), world);
      const double max_ms = all_ms[static_cast<std::size_t>(slow)];
      const double seconds = max_ms / 1.0e3;
      std::printf("device: %s\n", name.c_str());
      std::printf("ranks: %d  tokens: %d  hidden: %d  topk: %d  experts_per_rank: %d\n", world,
                  tokens, hidden, topk, experts);
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
  } catch (const sycl::exception &ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return EXIT_FAILURE;
  } catch (const std::bad_alloc &) {
    std::fprintf(stderr, "device allocation failed\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return EXIT_FAILURE;
  }
}
