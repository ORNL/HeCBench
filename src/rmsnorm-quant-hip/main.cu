#include "common.h"
#include "fp8_e4m3.h"
#include "reference.h"
#include "reduce.cuh"
#include "utils.cuh"
#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <hip/hip_fp8.h>

// vLLM fused RMSNorm + static FP8:
//   * invert the per-tensor scale once and multiply
//   * 128-bit vector loads/stores (vec4 float)
//   * single global read of the row when it fits in registers
//   * smaller blocks when many tokens so more SMs stay busy

constexpr int kVec = 4;
constexpr int kMaxVpt = 16;   // max vecs/thread; 1024 * 4 * 16 = 65536 hidden

// NATIVE: hardware E4M3 on gfx940/941/942 (FNUZ) and gfx950/1200/1201/1250 (OCP).
// Software rq::scaled_fp8 is used for correctness checks so they match the CPU.
template <bool FNUZ, bool NATIVE>
__device__ __forceinline__ uint8_t quant_fp8(float val, float scale_inv) {
  if constexpr (NATIVE) {
    if constexpr (FNUZ) {
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
      float x = val * scale_inv;
      x = fmaxf(-240.0f, fminf(x, 240.0f));
      return __hip_fp8_e4m3_fnuz(x).__x;
#else
      return rq::scaled_fp8(val, scale_inv, true, true);
#endif
    } else {
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__) || \
     defined(__gfx1250__))
      float x = val * scale_inv;
      x = fmaxf(-448.0f, fminf(x, 448.0f));
      return __hip_fp8_e4m3(x).__x;
#else
      return rq::scaled_fp8(val, scale_inv, true, false);
#endif
    }
  } else {
    return rq::scaled_fp8(val, scale_inv, true, FNUZ);
  }
}

template <bool FNUZ, bool NATIVE, int THREADS_PER_WARP>
__global__ void rmsnorm_quant_two_pass(const float *__restrict__ input,
                                       const float *__restrict__ gamma,
                                       uint8_t *__restrict__ output,
                                       float scale_inv, int64_t inner_len,
                                       int64_t outer_len, float epsilon) {
  constexpr int VEC = kVec;
  const int BLOCKSIZE = blockDim.x;
  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const int lane_id = threadIdx.x % THREADS_PER_WARP;
  const int start_offset =
      warp_id * THREADS_PER_WARP * VEC + lane_id * VEC;

  // Persistent row loop (Triton fused-softmax): programs stride by gridDim.x.
  for (int64_t bid = blockIdx.x; bid < outer_len; bid += gridDim.x) {
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
        BlockReduce<SumOp, float, THREADS_PER_WARP>(local_squares_sum) /
        static_cast<float>(inner_len);
    const float norm_factor = rsqrtf(mean_square + epsilon);

    float ld_gamma_regs[VEC];
    uint8_t st_regs[VEC];
    for (int64_t offset = start_offset; offset < inner_len;
         offset += static_cast<int64_t>(BLOCKSIZE) * VEC) {
      load_data<float, VEC>(input_ptr + offset, ld_input_regs);
      load_data<float, VEC>(gamma_ptr + offset, ld_gamma_regs);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const float val = ld_input_regs[i] * norm_factor * ld_gamma_regs[i];
        st_regs[i] = quant_fp8<FNUZ, NATIVE>(val, scale_inv);
      }
      store_data<uint8_t, VEC>(output_ptr + offset, st_regs);
    }
  }
}

// Single global read of the row into registers (vLLM layernorm_single_read.cuh).
template <bool FNUZ, bool NATIVE, int THREADS_PER_WARP>
__global__ void rmsnorm_quant_single_read(const float *__restrict__ input,
                                          const float *__restrict__ gamma,
                                          uint8_t *__restrict__ output,
                                          float scale_inv, int64_t inner_len,
                                          int64_t outer_len, float epsilon,
                                          int vpt) {
  constexpr int VEC = kVec;
  const int tid = threadIdx.x;

  for (int64_t bid = blockIdx.x; bid < outer_len; bid += gridDim.x) {
    const float *input_ptr = input + bid * inner_len;
    const float *gamma_ptr = gamma;
    uint8_t *output_ptr = output + bid * inner_len;

    float x[kMaxVpt * VEC];
    float w[kMaxVpt * VEC];
    float ss = 0.0f;

#pragma unroll 1
    for (int v = 0; v < vpt; ++v) {
      const int idx = (v * blockDim.x + tid) * VEC;
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
        BlockReduce<SumOp, float, THREADS_PER_WARP>(ss) /
        static_cast<float>(inner_len);
    const float norm_factor = rsqrtf(mean_square + epsilon);

#pragma unroll 1
    for (int v = 0; v < vpt; ++v) {
      uint8_t st[VEC];
      const int idx = (v * blockDim.x + tid) * VEC;
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const float val = x[v * VEC + j] * norm_factor * w[v * VEC + j];
        st[j] = quant_fp8<FNUZ, NATIVE>(val, scale_inv);
      }
      store_data<uint8_t, VEC>(output_ptr + idx, st);
    }
  }
}

struct LaunchPlan {
  bool single;
  int vpt;
  unsigned int grid;
};

static unsigned int persistent_grid(int64_t outer_len, int blocks_per_sm,
                                    int num_sm) {
  if (blocks_per_sm < 1)
    blocks_per_sm = 1;
  int64_t nprog = static_cast<int64_t>(num_sm) * blocks_per_sm;
  if (nprog > outer_len)
    nprog = outer_len;
  if (nprog < 1)
    nprog = 1;
  return static_cast<unsigned int>(nprog);
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

template <bool FNUZ, bool NATIVE, int THREADS_PER_WARP>
static LaunchPlan make_plan(int64_t inner_len, int64_t outer_len, int block_size) {
  LaunchPlan plan;
  plan.single = kernel_kind(inner_len, block_size, &plan.vpt);
  if (!plan.single && (inner_len % kVec) != 0) {
    fprintf(stderr, "inner length %ld must be a multiple of %d\n",
            (long)inner_len, kVec);
    exit(EXIT_FAILURE);
  }

  int device = 0;
  hipCheck(hipGetDevice(&device));
  int num_sm = 0;
  hipCheck(hipDeviceGetAttribute(&num_sm, hipDeviceAttributeMultiprocessorCount,
                                 device));
  int blocks_per_sm = 0;
  if (plan.single)
    hipCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, rmsnorm_quant_single_read<FNUZ, NATIVE, THREADS_PER_WARP>,
        block_size, 0));
  else
    hipCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, rmsnorm_quant_two_pass<FNUZ, NATIVE, THREADS_PER_WARP>,
        block_size, 0));
  plan.grid = persistent_grid(outer_len, blocks_per_sm, num_sm);
  return plan;
}

template <bool FNUZ, bool NATIVE, int THREADS_PER_WARP>
static void rmsnorm_quant_forward(const float *input, const float *gamma,
                                  uint8_t *output, float scale_inv,
                                  int64_t inner_len, int64_t outer_len,
                                  float epsilon, int block_size,
                                  const LaunchPlan &plan) {
  const dim3 block_dim(block_size, 1, 1);
  const dim3 grid_dim(plan.grid, 1, 1);
  if (plan.single)
    rmsnorm_quant_single_read<FNUZ, NATIVE, THREADS_PER_WARP>
        <<<grid_dim, block_dim>>>(input, gamma, output, scale_inv, inner_len,
                                  outer_len, epsilon, plan.vpt);
  else
    rmsnorm_quant_two_pass<FNUZ, NATIVE, THREADS_PER_WARP>
        <<<grid_dim, block_dim>>>(input, gamma, output, scale_inv, inner_len,
                                  outer_len, epsilon);
}

static double traffic_bytes(size_t size, int H, bool single) {
  const double input = (double)size * sizeof(float);
  const double gamma = (double)H * sizeof(float);
  const double out = (double)size;
  return (single ? input : 2.0 * input) + gamma + out;
}

static hipDeviceProp_t hip_device_prop() {
  int device = 0;
  hipDeviceProp_t prop;
  hipCheck(hipGetDevice(&device));
  hipCheck(hipGetDeviceProperties(&prop, device));
  return prop;
}

static bool hip_arch_uses_fnuz(const hipDeviceProp_t &prop) {
  const std::string arch(prop.gcnArchName);
  return arch.find("gfx940") != std::string::npos ||
         arch.find("gfx941") != std::string::npos ||
         arch.find("gfx942") != std::string::npos;
}

template <typename F>
static void with_fnuz_warp(bool fnuz, int warpSize, F &&f) {
  if (warpSize != 32 && warpSize != 64) {
    fprintf(stderr, "unsupported wave size %d\n", warpSize);
    exit(EXIT_FAILURE);
  }
  if (fnuz) {
    if (warpSize == 64)
      f(std::true_type{}, std::integral_constant<int, 64>{});
    else
      f(std::true_type{}, std::integral_constant<int, 32>{});
  } else if (warpSize == 64) {
    f(std::false_type{}, std::integral_constant<int, 64>{});
  } else {
    f(std::false_type{}, std::integral_constant<int, 32>{});
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <rows> <cols> <repeat>\n", argv[0]);
    return 1;
  }
  const long N = atol(argv[1]);
  const int H = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  const hipDeviceProp_t prop = hip_device_prop();
  const bool fnuz = hip_arch_uses_fnuz(prop);
  const int warp_size = prop.warpSize;

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

  srand(0);
  const size_t size = N * H;
  float *inp = make_random_float(size);
  float *gamma = make_random_float(H);

  float amax = 0.0f;
  for (size_t i = 0; i < size; i++)
    amax = fmaxf(amax, fabsf(inp[i]));
  // Static per-tensor scale from input amax (caller-supplied in vLLM).
  // RMSNorm can grow slightly; 2x headroom keeps values in range.
  const float qmax = rq::fp8_max(fnuz);
  const float scale = (amax > 0.0f) ? (2.0f * amax / qmax) : 1.0f;
  const float scale_inv = 1.0f / scale;

  uint8_t *out_ref = (uint8_t *)malloc(size);
  rmsnorm_quant_cpu(out_ref, inp, gamma, scale_inv, true, fnuz, N, H);

  float *d_inp, *d_gamma;
  uint8_t *d_out;
  hipCheck(hipMalloc(&d_inp, size * sizeof(float)));
  hipCheck(hipMalloc(&d_gamma, H * sizeof(float)));
  hipCheck(hipMalloc(&d_out, size));
  hipCheck(hipMemcpy(d_inp, inp, size * sizeof(float), hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_gamma, gamma, H * sizeof(float), hipMemcpyHostToDevice));

  printf("FP8 E4M3 format: %s  scale=%.6g (inverted)\n", fnuz ? "FNUZ" : "OCP",
         scale);

  int block_sizes[] = {32, 64, 128, 256, 512, 1024};

  uint8_t *h_out = (uint8_t *)malloc(size);

  auto launch = [&](bool native, int block_size) {
    with_fnuz_warp(fnuz, warp_size, [&](auto fnuz_t, auto warp_t) {
      constexpr bool F = decltype(fnuz_t)::value;
      constexpr int W = decltype(warp_t)::value;
      if (native) {
        const LaunchPlan plan = make_plan<F, true, W>(H, N, block_size);
        rmsnorm_quant_forward<F, true, W>(d_inp, d_gamma, d_out, scale_inv, H, N,
                                          1e-5f, block_size, plan);
      } else {
        const LaunchPlan plan = make_plan<F, false, W>(H, N, block_size);
        rmsnorm_quant_forward<F, false, W>(d_inp, d_gamma, d_out, scale_inv, H,
                                           N, 1e-5f, block_size, plan);
      }
    });
  };

  for (const auto &block_size : block_sizes) {
    if (block_size % warp_size != 0) {
      printf("Skipping block size %d (not a multiple of warp size %d).\n",
             block_size, warp_size);
      continue;
    }
    printf("Checking block size %d (software FP8).\n", block_size);
    launch(false, block_size);
    hipCheck(hipMemcpy(h_out, d_out, size, hipMemcpyDeviceToHost));
    if (!check_fp8_vs_cpu(h_out, out_ref, size, fnuz, 1))
      return 1;
  }

  // Hardware cvt may differ from software round-to-even by a couple of codes.
  bool bench_native = true;
  for (const auto &block_size : block_sizes) {
    if (block_size % warp_size != 0)
      continue;
    printf("Checking block size %d (native FP8).\n", block_size);
    launch(true, block_size);
    hipCheck(hipMemcpy(h_out, d_out, size, hipMemcpyDeviceToHost));
    if (!check_fp8_vs_cpu(h_out, out_ref, size, fnuz, 2)) {
      printf("Native FP8 does not match the CPU reference; "
             "benchmarking the software path.\n");
      bench_native = false;
      break;
    }
  }

  if (bench_native)
    printf("All results match. Starting benchmarks (native FP8).\n\n");
  else
    printf("Starting benchmarks (software FP8).\n\n");

  for (const auto &block_size : block_sizes) {
    if (block_size % warp_size != 0)
      continue;
    LaunchPlan plan;
    with_fnuz_warp(fnuz, warp_size, [&](auto fnuz_t, auto warp_t) {
      constexpr bool F = decltype(fnuz_t)::value;
      constexpr int W = decltype(warp_t)::value;
      if (bench_native)
        plan = make_plan<F, true, W>(H, N, block_size);
      else
        plan = make_plan<F, false, W>(H, N, block_size);
    });
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeat; r++) {
      with_fnuz_warp(fnuz, warp_size, [&](auto fnuz_t, auto warp_t) {
        constexpr bool F = decltype(fnuz_t)::value;
        constexpr int W = decltype(warp_t)::value;
        if (bench_native)
          rmsnorm_quant_forward<F, true, W>(d_inp, d_gamma, d_out, scale_inv, H,
                                            N, 1e-5f, block_size, plan);
        else
          rmsnorm_quant_forward<F, false, W>(d_inp, d_gamma, d_out, scale_inv, H,
                                             N, 1e-5f, block_size, plan);
      });
    }
    hipCheck(hipDeviceSynchronize());
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
  hipCheck(hipFree(d_out));
  hipCheck(hipFree(d_inp));
  hipCheck(hipFree(d_gamma));
  return 0;
}
