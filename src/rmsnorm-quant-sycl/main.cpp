#include "common.h"
#include "fp8_e4m3.h"
#include "reference.h"
#include "utils.h"
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <type_traits>

namespace syclex = sycl::ext::oneapi::experimental;

template <int THREADS_PER_WARP, bool FNUZ> class rmsnorm_quant_two_pass_k;
template <int THREADS_PER_WARP, bool FNUZ> class rmsnorm_quant_single_read_k;

constexpr int kVec = 4;
constexpr int kMaxVpt = 16;   // max vecs/thread; 1024 * 4 * 16 = 65536 hidden

template <int THREADS_PER_WARP, bool FNUZ>
void rmsnorm_quant_two_pass(const float *__restrict__ input,
                            const float *__restrict__ gamma,
                            uint8_t *__restrict__ output, float scale_inv,
                            int64_t inner_len, int64_t outer_len, float epsilon,
                            sycl::nd_item<3> &item) {
  constexpr int VEC = kVec;
  const int BLOCKSIZE = item.get_local_range(2);
  const int warp_id = item.get_local_id(2) / THREADS_PER_WARP;
  const int lane_id = item.get_local_id(2) % THREADS_PER_WARP;
  const int start_offset =
      warp_id * THREADS_PER_WARP * VEC + lane_id * VEC;
  const int64_t num_programs = item.get_group_range(2);

  // Persistent row loop (Triton fused-softmax): programs stride by grid size.
  for (int64_t bid = item.get_group(2); bid < outer_len; bid += num_programs) {
    const float *input_ptr = input + bid * inner_len;
    const float *gamma_ptr = gamma;
    uint8_t *output_ptr = output + bid * inner_len;

    float local_squares_sum = 0.0f;
    float ld_input_regs[VEC];

    for (int64_t offset = start_offset; offset < inner_len;
         offset += static_cast<int64_t>(BLOCKSIZE) * VEC) {
      load_data<float, VEC>(input_ptr + offset, ld_input_regs);
#pragma unroll
      for (int i = 0; i < VEC; ++i)
        local_squares_sum += ld_input_regs[i] * ld_input_regs[i];
    }

    const float mean_square =
        sycl::reduce_over_group(item.get_group(), local_squares_sum,
                                sycl::plus<float>()) /
        static_cast<float>(inner_len);
    const float norm_factor = sycl::rsqrt(mean_square + epsilon);

    float ld_gamma_regs[VEC];
    uint8_t st_regs[VEC];
    for (int64_t offset = start_offset; offset < inner_len;
         offset += static_cast<int64_t>(BLOCKSIZE) * VEC) {
      load_data<float, VEC>(input_ptr + offset, ld_input_regs);
      load_data<float, VEC>(gamma_ptr + offset, ld_gamma_regs);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const float val = ld_input_regs[i] * norm_factor * ld_gamma_regs[i];
        st_regs[i] = rq::scaled_fp8(val, scale_inv, true, FNUZ);
      }
      store_data<uint8_t, VEC>(output_ptr + offset, st_regs);
    }
  }
}

// Single global read of the row into registers (vLLM layernorm_single_read.cuh).
template <int THREADS_PER_WARP, bool FNUZ>
void rmsnorm_quant_single_read(const float *__restrict__ input,
                               const float *__restrict__ gamma,
                               uint8_t *__restrict__ output, float scale_inv,
                               int64_t inner_len, int64_t outer_len,
                               float epsilon, int vpt, sycl::nd_item<3> &item) {
  constexpr int VEC = kVec;
  const int tid = item.get_local_id(2);
  const int block_size = item.get_local_range(2);
  const int64_t num_programs = static_cast<int64_t>(item.get_group_range(2));

  for (int64_t bid = item.get_group(2); bid < outer_len; bid += num_programs) {
    const float *input_ptr = input + bid * inner_len;
    const float *gamma_ptr = gamma;
    uint8_t *output_ptr = output + bid * inner_len;

    float x[kMaxVpt * VEC];
    float w[kMaxVpt * VEC];
    float ss = 0.0f;

#pragma unroll 1
    for (int v = 0; v < vpt; ++v) {
      const int idx = (v * block_size + tid) * VEC;
      float in_v[VEC];
      float w_v[VEC];
      load_data<float, VEC>(input_ptr + idx, in_v);
      load_data<float, VEC>(gamma_ptr + idx, w_v);
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        x[v * VEC + j] = in_v[j];
        w[v * VEC + j] = w_v[j];
        ss += in_v[j] * in_v[j];
      }
    }

    const float mean_square =
        sycl::reduce_over_group(item.get_group(), ss, sycl::plus<float>()) /
        static_cast<float>(inner_len);
    const float norm_factor = sycl::rsqrt(mean_square + epsilon);

#pragma unroll 1
    for (int v = 0; v < vpt; ++v) {
      uint8_t st[VEC];
      const int idx = (v * block_size + tid) * VEC;
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const float val = x[v * VEC + j] * norm_factor * w[v * VEC + j];
        st[j] = rq::scaled_fp8(val, scale_inv, true, FNUZ);
      }
      store_data<uint8_t, VEC>(output_ptr + idx, st);
    }
  }
}

struct LaunchPlan {
  bool single;
  int vpt;
  size_t nprog;
};

static size_t persistent_grid(int64_t outer_len, int64_t max_wgs) {
  if (max_wgs > outer_len)
    max_wgs = outer_len;
  if (max_wgs < 1)
    max_wgs = 1;
  return static_cast<size_t>(max_wgs);
}

template <typename KernelName>
static size_t occupancy_grid(sycl::queue &q, int64_t outer_len,
                             const sycl::range<3> &lws, const char *kind,
                             int block_size) {
  const int64_t cus =
      q.get_device().get_info<sycl::info::device::max_compute_units>();
  uint32_t max_wgs = 0;
  const char *src = "fallback-cus";
  try {
    auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        q.get_context(), {q.get_device()});
    if (bundle.template has_kernel<KernelName>()) {
      auto kernel = bundle.template get_kernel<KernelName>();
      // Concurrent work-groups (intel/llvm#22667 HIP adapter): occupancy × CUs.
      max_wgs = kernel.template ext_oneapi_get_info<
          syclex::info::kernel_queue_specific::max_num_work_groups>(q, lws, 0);
      src = "max_num_work_groups";
    }
  } catch (const sycl::exception &e) {
    fprintf(stderr, "occupancy query failed: %s\n", e.what());
    src = "exception";
  }
  // Only replace a zero result. A small non-zero value is occupancy, not the
  // old HIP 0/1 stub (intel/llvm#21803 / #22667).
  if (max_wgs == 0) {
    max_wgs = static_cast<uint32_t>(cus);
    src = "fallback-cus";
  }
  const size_t nprog = persistent_grid(outer_len, static_cast<int64_t>(max_wgs));
  printf("  occupancy %s block %d: CUs=%ld query=%u (%s) nprog=%zu\n", kind,
         block_size, (long)cus, max_wgs, src, nprog);
  return nprog;
}

static bool kernel_kind(int64_t inner_len, int block_size, int *vpt_out) {
  const bool vec_ok = (inner_len % kVec) == 0;
  const int vpt =
      vec_ok ? static_cast<int>(inner_len / (static_cast<int64_t>(block_size) * kVec))
             : 0;
  *vpt_out = vpt;
  return vec_ok && vpt >= 1 && vpt <= kMaxVpt &&
         (inner_len == static_cast<int64_t>(block_size) * kVec * vpt);
}

template <int THREADS_PER_WARP, bool FNUZ>
static LaunchPlan make_plan(sycl::queue &q, int64_t inner_len, int64_t outer_len,
                            int block_size) {
  LaunchPlan plan;
  plan.single = kernel_kind(inner_len, block_size, &plan.vpt);
  if (!plan.single && (inner_len % kVec) != 0) {
    fprintf(stderr, "inner length %ld must be a multiple of %d\n",
            (long)inner_len, kVec);
    exit(EXIT_FAILURE);
  }
  const sycl::range<3> lws(1, 1, block_size);
  if (plan.single)
    plan.nprog =
        occupancy_grid<rmsnorm_quant_single_read_k<THREADS_PER_WARP, FNUZ>>(
            q, outer_len, lws, "single-read", block_size);
  else
    plan.nprog = occupancy_grid<rmsnorm_quant_two_pass_k<THREADS_PER_WARP, FNUZ>>(
        q, outer_len, lws, "two-pass", block_size);
  return plan;
}

template <int THREADS_PER_WARP, bool FNUZ>
void rmsnorm_quant_forward(sycl::queue &q, const float *input, const float *gamma,
                           uint8_t *output, float scale_inv, int64_t inner_len,
                           int64_t outer_len, float epsilon, int block_size,
                           const LaunchPlan &plan) {
  const sycl::range<3> lws(1, 1, block_size);
  const sycl::range<3> gws(1, 1, plan.nprog * static_cast<size_t>(block_size));
  const int vpt = plan.vpt;

  if (plan.single) {
    q.parallel_for<rmsnorm_quant_single_read_k<THREADS_PER_WARP, FNUZ>>(
        sycl::nd_range<3>(gws, lws),
        [=](sycl::nd_item<3> item)
            [[sycl::reqd_sub_group_size(THREADS_PER_WARP)]] {
              rmsnorm_quant_single_read<THREADS_PER_WARP, FNUZ>(
                  input, gamma, output, scale_inv, inner_len, outer_len, epsilon,
                  vpt, item);
            });
  } else {
    q.parallel_for<rmsnorm_quant_two_pass_k<THREADS_PER_WARP, FNUZ>>(
        sycl::nd_range<3>(gws, lws),
        [=](sycl::nd_item<3> item)
            [[sycl::reqd_sub_group_size(THREADS_PER_WARP)]] {
              rmsnorm_quant_two_pass<THREADS_PER_WARP, FNUZ>(
                  input, gamma, output, scale_inv, inner_len, outer_len, epsilon,
                  item);
            });
  }
}

static double traffic_bytes(size_t size, int H, bool single) {
  const double input = (double)size * sizeof(float);
  const double gamma = (double)H * sizeof(float);
  const double out = (double)size;
  return (single ? input : 2.0 * input) + gamma + out;
}

// MI300 (gfx940/941/942) uses AMD E4M3FNUZ; later AMD GPUs and other
// vendors use OCP E4M3. Same ISA list as HIP's architecture check.
static bool device_uses_fnuz(const sycl::device &dev) {
  try {
    const auto arch =
        dev.get_info<syclex::info::device::architecture>();
    if (arch == syclex::architecture::amd_gpu_gfx940 ||
        arch == syclex::architecture::amd_gpu_gfx941 ||
        arch == syclex::architecture::amd_gpu_gfx942)
      return true;
  } catch (const sycl::exception &) {
  }
  // HIP uses gcnArchName (gfx940/941/942). Some runtimes do not expose the
  // architecture query, but identify this FNUZ family as MI300.
  try {
    const auto name = dev.get_info<sycl::info::device::name>();
    if (name.find("gfx940") != std::string::npos ||
        name.find("gfx941") != std::string::npos ||
        name.find("gfx942") != std::string::npos ||
        name.find("MI300") != std::string::npos)
      return true;
  } catch (const sycl::exception &) {
  }
  return false;
}

template <typename F>
static void with_threads_per_warp(int warpSize, F &&f) {
  switch (warpSize) {
  case 8:
    f(std::integral_constant<int, 8>{});
    break;
  case 16:
    f(std::integral_constant<int, 16>{});
    break;
  case 32:
    f(std::integral_constant<int, 32>{});
    break;
  case 64:
    f(std::integral_constant<int, 64>{});
    break;
  default:
    fprintf(stderr, "unsupported wave size %d\n", warpSize);
    exit(EXIT_FAILURE);
  }
}

template <typename F>
static void with_fnuz_warp(bool fnuz, int warpSize, F &&f) {
  with_threads_per_warp(warpSize, [&](auto w) {
    if (fnuz)
      f(std::true_type{}, w);
    else
      f(std::false_type{}, w);
  });
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <rows> <cols> <repeat>\n", argv[0]);
    return 1;
  }
  const long N = atol(argv[1]);
  const int H = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  if (N <= 0 || H <= 0 || repeat <= 0) {
    fprintf(stderr, "rows, cols and repeat must be positive\n");
    return 1;
  }
  if (H % kVec != 0) {
    fprintf(stderr, "cols must be a multiple of %d\n", kVec);
    return 1;
  }
  if (static_cast<size_t>(H) > SIZE_MAX / static_cast<size_t>(N)) {
    fprintf(stderr, "rows * cols overflows size_t\n");
    return 1;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  const bool fnuz = device_uses_fnuz(q.get_device());

  srand(0);
  const size_t size = N * H;
  float *inp = make_random_float(size);
  float *gamma = make_random_float(H);

  float amax = 0.0f;
  for (size_t i = 0; i < size; i++)
    amax = fmaxf(amax, fabsf(inp[i]));
  const float qmax = rq::fp8_max(fnuz);
  const float scale = (amax > 0.0f) ? (2.0f * amax / qmax) : 1.0f;
  const float scale_inv = 1.0f / scale;

  uint8_t *out_ref = (uint8_t *)malloc(size);
  rmsnorm_quant_cpu(out_ref, inp, gamma, scale_inv, true, fnuz, N, H);

  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (sg_sizes.empty()) {
    fprintf(stderr, "device reports no sub-group sizes\n");
    return 1;
  }
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  const int warpSize = *r;

  float *d_inp = sycl::malloc_device<float>(size, q);
  float *d_gamma = sycl::malloc_device<float>(H, q);
  uint8_t *d_out = sycl::malloc_device<uint8_t>(size, q);
  q.memcpy(d_inp, inp, size * sizeof(float));
  q.memcpy(d_gamma, gamma, H * sizeof(float));
  q.wait();

  printf("FP8 E4M3 format: %s  scale=%.6g (inverted)\n", fnuz ? "FNUZ" : "OCP",
         scale);

  int block_sizes[] = {32, 64, 128, 256, 512, 1024};

  uint8_t *h_out = (uint8_t *)malloc(size);

  auto launch = [&](int block_size, const LaunchPlan &plan) {
    with_fnuz_warp(fnuz, warpSize, [&](auto fnuz_t, auto w) {
      constexpr bool F = decltype(fnuz_t)::value;
      constexpr int W = decltype(w)::value;
      rmsnorm_quant_forward<W, F>(q, d_inp, d_gamma, d_out, scale_inv, H, N,
                                  1e-5f, block_size, plan);
    });
  };

  auto plan_for = [&](int block_size) {
    LaunchPlan plan;
    with_fnuz_warp(fnuz, warpSize, [&](auto fnuz_t, auto w) {
      constexpr bool F = decltype(fnuz_t)::value;
      constexpr int W = decltype(w)::value;
      plan = make_plan<W, F>(q, H, N, block_size);
    });
    return plan;
  };

  for (const auto &block_size : block_sizes) {
    if (block_size % warpSize != 0) {
      printf("Skipping block size %d (not a multiple of warp size %d).\n",
             block_size, warpSize);
      continue;
    }
    printf("Checking block size %d.\n", block_size);
    const LaunchPlan plan = plan_for(block_size);
    launch(block_size, plan);
    q.memcpy(h_out, d_out, size).wait();
    if (!check_fp8_vs_cpu(h_out, out_ref, size, fnuz, 1))
      return 1;
  }

  printf("All results match. Starting benchmarks.\n\n");

  for (const auto &block_size : block_sizes) {
    if (block_size % warpSize != 0)
      continue;
    const LaunchPlan plan = plan_for(block_size);
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeat; r++)
      launch(block_size, plan);
    q.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;
    const float elapsed_time = duration.count() / repeat;
    const float memory_bandwidth =
        traffic_bytes(size, H, plan.single) / elapsed_time / 1e6;
    printf("block_size %4d | %s | time %.4f ms | bandwidth %.2f GB/s\n",
           block_size, plan.single ? "single-read" : "two-pass", elapsed_time,
           memory_bandwidth);
  }

  free(out_ref);
  free(h_out);
  free(inp);
  free(gamma);
  sycl::free(d_out, q);
  sycl::free(d_inp, q);
  sycl::free(d_gamma, q);
  return 0;
}
