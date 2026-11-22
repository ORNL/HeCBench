#include "common.h"
#include "fp8_e4m3.h"
#include "reference.h"
#include "reduce.cuh"
#include "utils.cuh"
#include <cstdint>
#include <cstdlib>

// vLLM fused RMSNorm + static FP8:
//   * invert the per-tensor scale once and multiply
//   * 128-bit vector loads/stores (vec4 float)
//   * single global read of the row when it fits in registers
//   * smaller blocks when many tokens so more SMs stay busy

constexpr int kVec = 4;       // 16-byte loads
constexpr int kMaxVpt = 16;   // max vecs/thread; 1024 * 4 * 16 = 65536 hidden

__device__ __forceinline__ uint8_t quant_fp8(float val, float scale_inv) {
  return rq::scaled_fp8(val, scale_inv, true, false);
}

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
        BlockReduce<SumOp, float>(local_squares_sum) /
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
        st_regs[i] = quant_fp8(val, scale_inv);
      }
      store_data<uint8_t, VEC>(output_ptr + offset, st_regs);
    }
  }
}

// Single global read of the row into registers (vLLM layernorm_single_read.cuh).
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
        BlockReduce<SumOp, float>(ss) / static_cast<float>(inner_len);
    const float norm_factor = rsqrtf(mean_square + epsilon);

#pragma unroll 1
    for (int v = 0; v < vpt; ++v) {
      uint8_t st[VEC];
      const int idx = (v * blockDim.x + tid) * VEC;
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const float val = x[v * VEC + j] * norm_factor * w[v * VEC + j];
        st[j] = quant_fp8(val, scale_inv);
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

static LaunchPlan make_plan(int64_t inner_len, int64_t outer_len, int block_size) {
  LaunchPlan plan;
  plan.single = kernel_kind(inner_len, block_size, &plan.vpt);
  if (!plan.single && (inner_len % kVec) != 0) {
    fprintf(stderr, "inner length %ld must be a multiple of %d\n",
            (long)inner_len, kVec);
    exit(EXIT_FAILURE);
  }

  int device = 0;
  cudaCheck(cudaGetDevice(&device));
  int num_sm = 0;
  cudaCheck(
      cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_sm = 0;
  if (plan.single)
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, rmsnorm_quant_single_read, block_size, 0));
  else
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, rmsnorm_quant_two_pass, block_size, 0));
  plan.grid = persistent_grid(outer_len, blocks_per_sm, num_sm);
  return plan;
}

static void rmsnorm_quant_forward(const float *input, const float *gamma,
                                  uint8_t *output, float scale_inv,
                                  int64_t inner_len, int64_t outer_len,
                                  float epsilon, int block_size,
                                  const LaunchPlan &plan) {
  const dim3 block_dim(block_size, 1, 1);
  const dim3 grid_dim(plan.grid, 1, 1);
  if (plan.single)
    rmsnorm_quant_single_read<<<grid_dim, block_dim>>>(
        input, gamma, output, scale_inv, inner_len, outer_len, epsilon,
        plan.vpt);
  else
    rmsnorm_quant_two_pass<<<grid_dim, block_dim>>>(
        input, gamma, output, scale_inv, inner_len, outer_len, epsilon);
}

// two-pass reads input twice; gamma and fp8 output once.
static double traffic_bytes(size_t size, int H, bool single) {
  const double input = (double)size * sizeof(float);
  const double gamma = (double)H * sizeof(float);
  const double out = (double)size;
  return (single ? input : 2.0 * input) + gamma + out;
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

  srand(0);
  const size_t size = N * H;
  float *inp = make_random_float(size);
  float *gamma = make_random_float(H);

  float amax = 0.0f;
  for (size_t i = 0; i < size; i++)
    amax = fmaxf(amax, fabsf(inp[i]));
  // Static per-tensor scale from input amax (caller-supplied in vLLM).
  // RMSNorm can grow slightly; 2x headroom keeps values in range.
  const float qmax = rq::fp8_max(false);
  const float scale = (amax > 0.0f) ? (2.0f * amax / qmax) : 1.0f;
  const float scale_inv = 1.0f / scale;

  uint8_t *out_ref = (uint8_t *)malloc(size);
  rmsnorm_quant_cpu(out_ref, inp, gamma, scale_inv, true, false, N, H);

  float *d_inp, *d_gamma;
  uint8_t *d_out;
  cudaCheck(cudaMalloc(&d_inp, size * sizeof(float)));
  cudaCheck(cudaMalloc(&d_gamma, H * sizeof(float)));
  cudaCheck(cudaMalloc(&d_out, size));
  cudaCheck(cudaMemcpy(d_inp, inp, size * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_gamma, gamma, H * sizeof(float), cudaMemcpyHostToDevice));

  printf("FP8 E4M3 format: OCP  scale=%.6g (inverted)\n", scale);

  int block_sizes[] = {32, 64, 128, 256, 512, 1024};

  uint8_t *h_out = (uint8_t *)malloc(size);

  for (const auto &block_size : block_sizes) {
    const LaunchPlan plan = make_plan(H, N, block_size);
    printf("Checking block size %d.\n", block_size);
    rmsnorm_quant_forward(d_inp, d_gamma, d_out, scale_inv, H, N, 1e-5f,
                          block_size, plan);
    cudaCheck(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
    if (!check_fp8_vs_cpu(h_out, out_ref, size, false, 1))
      return 1;
  }

  printf("All results match. Starting benchmarks.\n\n");

  for (const auto &block_size : block_sizes) {
    const LaunchPlan plan = make_plan(H, N, block_size);
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeat; r++)
      rmsnorm_quant_forward(d_inp, d_gamma, d_out, scale_inv, H, N, 1e-5f,
                            block_size, plan);
    cudaCheck(cudaDeviceSynchronize());
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
  cudaCheck(cudaFree(d_out));
  cudaCheck(cudaFree(d_inp));
  cudaCheck(cudaFree(d_gamma));
  return 0;
}
