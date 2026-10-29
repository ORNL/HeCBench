// Mamba-2 SSD forward pass.
//
// cuTile kernels in NVIDIA TileGym (src/tilegym/ops/cutile/mamba2.py, MIT):
//   chunk_cumsum, fwd_pass_state, compute_out.
//
// The GPU evaluates the recurrence in three ordered stages:
//   1. chunk_cumsum converts a[h] * dt into cumulative base-2 decay exponents.
//   2. fwd_pass_state advances state chunk by chunk and records the state that
//      enters each chunk.
//   3. compute_out combines each chunk-entry state with its tokens and skip.
//
// Aligned 16-wide tiles use nvcuda::wmma. The GEMM path uses register-blocked
// 4x4 tiles. Its state update is
// X^T @ (dt * exp2(d_last - d) * B); its output uses C @ B^T plus the carried
// chunk-entry state. Shapes incompatible with those tiles use shared-memory
// tiled kernels or direct global-memory kernels.
//
// Data types follow TileGym's mamba2_chunk_forward; see ssm2_types.h.
//
// Chunks of one head remain sequential because they form a recurrence.
// Independent batches, heads, P/N tiles, and output chunks provide GPU
// parallelism. Kernel launch order on the default stream enforces stage order.
//
// Shapes:
//   x, out     [B, T, H, P]
//   dt, decay  [B, T, H]
//   a, dskip   [H]
//   b, c       [B, T, G, N]
//   state      [B, H, P, N]
//   hiddens    [B, H, nchunks, P, N]  (state entering each chunk)

#include <chrono>
#include <cuda_runtime.h>
#include <initializer_list>
#include <math.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ssm2_launch_policy.h"
#include "kernels.h"
#include "reference.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err_));                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Parse a positive command-line extent into the benchmark's supported range.
static int parse_positive(const char *text, const char *name, int *out) {
  char *end = nullptr;
  long value = strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0 || value > 1000000000L) {
    fprintf(stderr, "Invalid %s: %s\n", name, text);
    return 0;
  }
  *out = (int)value;
  return 1;
}

// Element counts use size_t. Extents themselves stay in int because each one
// is a grid or loop bound; only the flattened offset can exceed 2^31.
static bool mul_dims(size_t *out, std::initializer_list<int> dims) {
  size_t v = 1;
  for (int dim : dims) {
    if (dim <= 0)
      return false;
    const size_t d = (size_t)dim;
    if (v > SIZE_MAX / d)
      return false;
    v *= d;
  }
  *out = v;
  return true;
}

// Convert a validated element count to bytes without overflowing size_t.
static bool float_elems(size_t *out, std::initializer_list<int> dims) {
  size_t v = 0;
  if (!mul_dims(&v, dims) || v > SIZE_MAX / sizeof(float))
    return false;
  *out = v;
  return true;
}

// Cap a logical block count at the device's legal one-dimensional grid size.
static unsigned grid_blocks(size_t blocks, int max_grid_x) {
  const size_t limit = (size_t)max_grid_x;
  return (unsigned)(blocks < limit ? blocks : limit);
}

// TileGym's forward test tolerances (atol = rtol) against its FP32 reference.
static constexpr float kOutTol = 2e-2f;
static constexpr float kStateTol = 1e-2f;

// Apply torch.testing.assert_close's |got - ref| <= atol + rtol * |ref| test
// with atol = rtol = tol; non-finite values fail.
static bool all_close(const float *got, const float *ref, size_t n, float tol) {
  size_t bad = 0;
#pragma omp parallel for schedule(static) reduction(+ : bad)
  for (size_t i = 0; i < n; ++i) {
    const bool close = isfinite(got[i]) && isfinite(ref[i]) &&
                       fabsf(got[i] - ref[i]) <= tol + tol * fabsf(ref[i]);
    bad += !close;
  }
  return bad == 0;
}

// Round inputs to BF16 for the device and keep the rounded values in FP32 for
// the reference.
static void round_to_bf16(float *v, ssm2_bf16 *out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = ssm2_to_bf16(v[i]);
    v[i] = ssm2_from_bf16(out[i]);
  }
}

// Launch one complete SSD forward pass. Logical block counts use size_t and
// kernels grid-stride when a count exceeds the device's maximum grid x size.
static void launch_forward(const SSM2Params &p, bool use_mma, bool use_gemm,
                           bool smem_state, bool smem_out, int state_bh, int state_bs,
                           int out_bh, int state_splits, int max_grid_x,
                           size_t state_smem, size_t fwd_smem, size_t cb_smem,
                           size_t out_smem) {
  const size_t cumsum_blocks = (size_t)p.B * p.nchunks * p.H;
  chunk_cumsum_kernel<<<grid_blocks(cumsum_blocks, max_grid_x), p.chunk,
                        p.chunk * sizeof(float)>>>(p, cumsum_blocks);

  if (use_mma) {
    const size_t state_blocks =
        (size_t)p.B * p.H * (p.P / 16) * (p.N / 16);
    fwd_wmma_kernel<<<grid_blocks(state_blocks, max_grid_x), 32,
                       state_smem>>>(p, state_blocks);
    const size_t cb_blocks = (size_t)p.B * p.nchunks * p.G;
    cb_wmma_kernel<<<grid_blocks(cb_blocks, max_grid_x), 256>>>(p, cb_blocks);
    const size_t out_blocks = (size_t)p.B * p.H * p.nchunks *
                              (p.P / 16) * (p.chunk / 16);
    out_wmma_kernel<<<grid_blocks(out_blocks, max_grid_x), 32,
                       out_smem>>>(p, out_blocks);
    return;
  }

  if (use_gemm) {
    // Fast path: register-blocked state, C @ B^T, and output GEMMs.
    const int tile_threads = (state_bh / 4) * (state_bs / 4);
    const int state_threads = tile_threads * state_splits;
    const size_t state_blocks =
        (size_t)p.B * p.H * (p.P / state_bh) * (p.N / state_bs);
    fwd_gemm_kernel<<<grid_blocks(state_blocks, max_grid_x), state_threads,
                      fwd_smem>>>(p, state_bh, state_bs, state_blocks);
    const size_t cb_blocks = (size_t)p.B * p.nchunks * p.G;
    cb_gemm_kernel<<<grid_blocks(cb_blocks, max_grid_x),
                     (p.chunk / 4) * (p.chunk / 4), cb_smem>>>(p, cb_blocks);
    const int out_threads = (p.chunk / 4) * (out_bh / 4);
    const size_t out_blocks = (size_t)p.B * p.H * p.nchunks * (p.P / out_bh);
    compute_out_gemm_kernel<<<grid_blocks(out_blocks, max_grid_x), out_threads,
                              out_smem>>>(p, out_bh, out_blocks);
    return;
  }

  // Scalar state path: use P-by-N tiles when their staging buffers fit;
  // otherwise let each batch/head block stream the full state globally.
  if (smem_state) {
    const int ptiles = (p.P + kStatePTile - 1) / kStatePTile;
    const int ntiles = (p.N + kStateNTile - 1) / kStateNTile;
    const size_t state_blocks = (size_t)p.B * p.H * ptiles * ntiles;
    fwd_pass_state_tiled_kernel<<<grid_blocks(state_blocks, max_grid_x),
                                  kThreads, state_smem>>>(p, state_blocks);
  } else {
    const size_t state_blocks = (size_t)p.B * p.H;
    fwd_pass_state_global_kernel<<<grid_blocks(state_blocks, max_grid_x),
                                   kThreads>>>(p, state_blocks);
  }

  // Scalar output path: optionally materialize C @ B^T once per group/chunk.
  if (smem_out) {
    const size_t cb_blocks = (size_t)p.B * p.nchunks * p.G;
    compute_cb_kernel<<<grid_blocks(cb_blocks, max_grid_x), kThreads>>>(
        p, cb_blocks);
    const int ptiles = (p.P + kOutPTile - 1) / kOutPTile;
    const size_t out_blocks = (size_t)p.B * p.nchunks * p.H * ptiles;
    compute_out_tiled_kernel<<<grid_blocks(out_blocks, max_grid_x), kThreads,
                               out_smem>>>(p, out_blocks);
  } else {
    const size_t out_blocks = (size_t)p.B * p.nchunks * p.H;
    compute_out_global_kernel<<<grid_blocks(out_blocks, max_grid_x),
                                kThreads>>>(p, out_blocks);
  }
}

static const int kSweepChunks[] = {32, 64, 128};

struct ChunkPlan {
  bool use_mma;
  bool use_gemm;
  bool smem_state;
  bool smem_out;
  int state_bh, state_bs, out_bh, state_splits;
  size_t state_smem, fwd_smem, cb_smem, out_smem, cb_elems, hid_elems;
};

// Validate a chunk size and derive its kernel, workspace, and dynamic
// shared-memory settings. The shared launch policy adapts tiles and reduction
// splits to the problem shape and reported CUDA device properties.
static bool plan_chunk(int B, int T, int H, int G, int P, int N, int chunk,
                       int max_smem, bool mma_supported,
                       const SSM2DeviceInfo &device,
                       ChunkPlan *plan) {
  if (T % chunk != 0 || chunk > 1024)
    return false;
  const int nchunks = T / chunk;
  size_t hid_elems = 0, cb_elems = 0;
  if (!float_elems(&hid_elems, {B, H, nchunks, P, N}) ||
      !float_elems(&cb_elems, {B, nchunks, G, chunk, chunk}))
    return false;

  const SSM2LaunchPolicy launch =
      ssm2_select_launch_policy(B, T, H, P, N, chunk, device);
  const int state_bh = launch.state_bh;
  const int state_bs = launch.state_bs;
  const int out_bh = launch.out_bh;

  // Each GEMM thread computes a 4x4 fragment. Therefore C @ B^T uses one
  // thread per 4x4 chunk tile, output uses chunk/4 by out_bh/4 threads, and
  // state uses state_bh/4 by state_bs/4 threads.
  const int cb_threads = (chunk / 4) * (chunk / 4);
  const int out_threads = out_bh ? (chunk / 4) * (out_bh / 4) : 0;
  const int tile_threads =
      (state_bh && state_bs) ? (state_bh / 4) * (state_bs / 4) : 0;

  const int splits = launch.state_splits;
  const int state_threads = tile_threads * splits;

  // Shared memory holds FP32 copies of the BF16 operands. State stages X,
  // weighted B, and one 4x4 partial accumulator per state thread. C @ B^T
  // stages four N values from B and C per token. Output stages decay, C @ B^T,
  // dt * X, and four-wide C/state slices.
  const size_t fwd_smem =
      ((size_t)chunk * (state_bh + state_bs) + (size_t)state_threads * 16) *
      sizeof(float);
  const size_t cb_smem = (size_t)8 * chunk * sizeof(float);
  const size_t gemm_out_smem =
      (size_t)(chunk + chunk * chunk + chunk * out_bh + 4 * chunk +
               4 * out_bh) *
      sizeof(float);

  // The complete GEMM pipeline is atomic: all three stages must satisfy their
  // vectorization, block-size, and dynamic shared-memory constraints. If any
  // condition fails, planning selects scalar tiled/global kernels below.
  const bool use_gemm =
      chunk % 4 == 0 && chunk <= 128 && state_bh && state_bs && out_bh &&
      cb_threads > 0 && cb_threads <= 1024 && out_threads > 0 &&
      out_threads <= 1024 && state_threads > 0 && state_threads <= 1024 &&
      fwd_smem <= (size_t)max_smem && gemm_out_smem <= (size_t)max_smem;
  const size_t state_smem =
      (size_t)chunk * (kStatePTile + kStateNTile) * sizeof(float);
  const size_t tiled_out_smem =
      (size_t)(chunk + chunk * kOutPTile + chunk * chunk) * sizeof(float);
  // WMMA stages BF16 weighted B plus a 16x16 state tile, and FP32 dt * X plus
  // 16x16 accumulator and 16x8 TF32 operand tiles.
  const size_t mma_state_smem =
      (size_t)chunk * 16 * sizeof(ssm2_bf16) + 16 * 16 * sizeof(float);
  const size_t mma_out_smem =
      ((size_t)chunk * 16 + 16 * 16 + 16 * 8) * sizeof(float);
  plan->use_mma = mma_supported && chunk % 16 == 0 && P % 16 == 0 &&
                  N % 16 == 0 && mma_state_smem <= (size_t)max_smem &&
                  mma_out_smem <= (size_t)max_smem;
  plan->use_gemm = use_gemm && !plan->use_mma;

  // Scalar fallback selection is independent for state and output. Tiled
  // kernels are preferred when their fixed-size staging buffers fit; otherwise
  // the corresponding global-memory kernel requires no dynamic shared memory.
  plan->smem_state =
      !plan->use_mma && !plan->use_gemm && state_smem <= (size_t)max_smem;
  plan->smem_out =
      !plan->use_mma && !plan->use_gemm && tiled_out_smem <= (size_t)max_smem;
  plan->state_bh = state_bh;
  plan->state_bs = state_bs;
  plan->out_bh = out_bh;
  plan->state_splits = splits;
  plan->state_smem = plan->use_mma ? mma_state_smem : state_smem;
  plan->fwd_smem = fwd_smem;
  plan->cb_smem = cb_smem;
  plan->out_smem = plan->use_mma ? mma_out_smem
                                 : (use_gemm ? gemm_out_smem : tiled_out_smem);
  plan->cb_elems =
      (plan->use_mma || plan->use_gemm || plan->smem_out) ? cb_elems : 0;
  plan->hid_elems = hid_elems;
  return true;
}

// Parse the benchmark shape, execute each chunk plan, and validate GPU results.
int main(int argc, char **argv) {
  if (argc != 2 && argc != 8) {
    fprintf(
        stderr,
        "Usage: %s <repeat> [batch tokens heads groups head_size state_size]\n"
        "Defaults: repeat=100 B=2 T=256 H=8 G=2 P=32 N=64\n"
        "Sweeps chunk sizes 32, 64, and 128.\n",
        argv[0]);
    return 1;
  }

  int repeat = 100, B = 2, T = 256, H = 8, G = 2, P = 32, N = 64;
  if (!parse_positive(argv[1], "repeat", &repeat))
    return 1;
  if (argc == 8) {
    if (!parse_positive(argv[2], "batch", &B) ||
        !parse_positive(argv[3], "tokens", &T) ||
        !parse_positive(argv[4], "heads", &H) ||
        !parse_positive(argv[5], "groups", &G) ||
        !parse_positive(argv[6], "head_size", &P) ||
        !parse_positive(argv[7], "state_size", &N))
      return 1;
  }

  if (H % G != 0) {
    fprintf(stderr, "heads (%d) must be divisible by groups (%d)\n", H, G);
    return 1;
  }
  size_t x_elems = 0, dt_elems = 0, bc_elems = 0, st_elems = 0;
  if (!float_elems(&x_elems, {B, T, H, P}) ||
      !float_elems(&dt_elems, {B, T, H}) ||
      !float_elems(&bc_elems, {B, T, G, N}) ||
      !float_elems(&st_elems, {B, H, P, N})) {
    fprintf(stderr, "Problem size overflows size_t\n");
    return 1;
  }

  printf("ssm2 Mamba-2 SSD forward\n");
  printf("B=%d T=%d H=%d G=%d P=%d N=%d repeat=%d\n", B, T, H, G, P, N, repeat);
  printf("Chunk sweep:");
  for (int chunk : kSweepChunks)
    printf(" %d", chunk);
  printf("\n");

  // Host buffers hold deterministic inputs, the GPU result, and the scalar
  // oracle result. Element counts were checked before converting them to bytes.
  float *h_x = (float *)malloc(x_elems * sizeof(float));
  float *h_dt = (float *)malloc(dt_elems * sizeof(float));
  float *h_a = (float *)malloc((size_t)H * sizeof(float));
  float *h_d = (float *)malloc((size_t)H * sizeof(float));
  float *h_b = (float *)malloc(bc_elems * sizeof(float));
  float *h_c = (float *)malloc(bc_elems * sizeof(float));
  float *h_init = (float *)malloc(st_elems * sizeof(float));
  float *h_out = (float *)malloc(x_elems * sizeof(float));
  float *h_out_ref = (float *)malloc(x_elems * sizeof(float));
  float *h_state = (float *)malloc(st_elems * sizeof(float));
  float *h_state_ref = (float *)malloc(st_elems * sizeof(float));
  ssm2_bf16 *h_x16 = (ssm2_bf16 *)malloc(x_elems * sizeof(ssm2_bf16));
  ssm2_bf16 *h_dt16 = (ssm2_bf16 *)malloc(dt_elems * sizeof(ssm2_bf16));
  ssm2_bf16 *h_b16 = (ssm2_bf16 *)malloc(bc_elems * sizeof(ssm2_bf16));
  ssm2_bf16 *h_c16 = (ssm2_bf16 *)malloc(bc_elems * sizeof(ssm2_bf16));
  ssm2_bf16 *h_out16 = (ssm2_bf16 *)malloc(x_elems * sizeof(ssm2_bf16));
  if (!h_x || !h_dt || !h_a || !h_d || !h_b || !h_c || !h_init || !h_out ||
      !h_out_ref || !h_state || !h_state_ref || !h_x16 || !h_dt16 || !h_b16 ||
      !h_c16 || !h_out16) {
    fprintf(stderr, "Host allocation failed\n");
    return 1;
  }

  // Fixed-seed, bounded inputs make runs reproducible and keep the recurrent
  // state numerically stable.
  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> xdist(-0.5f, 0.5f);
  std::uniform_real_distribution<float> dt_dist(0.02f, 0.08f);
  std::uniform_real_distribution<float> a_dist(-0.8f, -0.1f);
  std::uniform_real_distribution<float> d_dist(-0.5f, 0.5f);
  std::uniform_real_distribution<float> s_dist(-0.1f, 0.1f);
  for (size_t i = 0; i < x_elems; ++i)
    h_x[i] = xdist(gen);
  for (size_t i = 0; i < dt_elems; ++i)
    h_dt[i] = dt_dist(gen);
  for (int i = 0; i < H; ++i) {
    h_a[i] = a_dist(gen);
    h_d[i] = d_dist(gen);
  }
  for (size_t i = 0; i < bc_elems; ++i) {
    h_b[i] = xdist(gen);
    h_c[i] = xdist(gen);
  }
  for (size_t i = 0; i < st_elems; ++i)
    h_init[i] = s_dist(gen);
  round_to_bf16(h_x, h_x16, x_elems);
  round_to_bf16(h_dt, h_dt16, dt_elems);
  round_to_bf16(h_b, h_b16, bc_elems);
  round_to_bf16(h_c, h_c16, bc_elems);

  // The reference performs O(B*T*H*P*N) scalar work. Skip impractically large
  // cases rather than letting validation dominate the GPU benchmark.
  const int ref_threads = omp_get_max_threads();
  // A couple of seconds of reference work on each thread. The Makefile shape
  // (B=8, T=2048, H=80, P=64, N=128) stays under this on one socket.
  const size_t ref_limit =
      (size_t)(ref_threads < 1 ? 1 : ref_threads) * 2000000000u;
  size_t ref_work = 0;
  const bool check =
      mul_dims(&ref_work, {B, T, H, P, N}) && ref_work <= ref_limit;
  if (check) {
    auto ref_start = std::chrono::steady_clock::now();
    ssm2_reference(h_x, h_a, h_b, h_c, h_d, h_dt, h_init, h_out_ref,
                   h_state_ref, B, T, H, G, P, N);
    auto ref_stop = std::chrono::steady_clock::now();
    printf("Reference (OpenMP, %d threads) time: %.3f ms\n", ref_threads,
           std::chrono::duration<double, std::milli>(ref_stop - ref_start)
               .count());
  } else if (ref_work > ref_limit) {
    printf("Reference skipped (B*T*H*P*N=%zu > %zu); validation will fail\n",
           ref_work, ref_limit);
  } else {
    printf("Reference skipped (B*T*H*P*N overflows size_t); validation will "
           "fail\n");
  }

  // Persistent device buffers are independent of chunk size. C @ B^T and
  // chunk-entry-state workspaces are allocated separately inside the sweep.
  ssm2_bf16 *d_x, *d_dt, *d_b, *d_c, *d_out;
  float *d_a, *d_d, *d_init, *d_decay, *d_state;
  CHECK(cudaMalloc(&d_x, x_elems * sizeof(ssm2_bf16)));
  CHECK(cudaMalloc(&d_dt, dt_elems * sizeof(ssm2_bf16)));
  CHECK(cudaMalloc(&d_a, (size_t)H * sizeof(float)));
  CHECK(cudaMalloc(&d_d, (size_t)H * sizeof(float)));
  CHECK(cudaMalloc(&d_b, bc_elems * sizeof(ssm2_bf16)));
  CHECK(cudaMalloc(&d_c, bc_elems * sizeof(ssm2_bf16)));
  CHECK(cudaMalloc(&d_init, st_elems * sizeof(float)));
  CHECK(cudaMalloc(&d_decay, dt_elems * sizeof(float)));

  CHECK(cudaMemcpy(d_x, h_x16, x_elems * sizeof(ssm2_bf16),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_dt, h_dt16, dt_elems * sizeof(ssm2_bf16),
                   cudaMemcpyHostToDevice));
  CHECK(
      cudaMemcpy(d_a, h_a, (size_t)H * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(
      cudaMemcpy(d_d, h_d, (size_t)H * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b16, bc_elems * sizeof(ssm2_bf16),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_c, h_c16, bc_elems * sizeof(ssm2_bf16),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_init, h_init, st_elems * sizeof(float),
                   cudaMemcpyHostToDevice));

  CHECK(cudaMalloc(&d_state, st_elems * sizeof(float)));
  CHECK(cudaMalloc(&d_out, x_elems * sizeof(ssm2_bf16)));

  // Bundle dimensions and device pointers so every kernel receives one
  // consistent description of the current problem and chunk plan.
  SSM2Params p;
  p.B = B;
  p.T = T;
  p.H = H;
  p.G = G;
  p.P = P;
  p.N = N;
  p.heads_per_group = H / G;
  p.x = d_x;
  p.a = d_a;
  p.b = d_b;
  p.c = d_c;
  p.dskip = d_d;
  p.dt = d_dt;
  p.init_state = d_init;
  p.decay = d_decay;
  p.final_state = d_state;
  p.out = d_out;

  // Planning uses reported occupancy-related limits; launch sizing separately
  // uses the device's legal one-dimensional grid limit.
  int max_smem = 0, max_grid_x = 0, compute_units = 0, compute_major = 0;
  int max_threads = 0, subgroup_width = 0;
  CHECK(cudaDeviceGetAttribute(&max_smem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  CHECK(cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, 0));
  CHECK(cudaDeviceGetAttribute(&compute_units, cudaDevAttrMultiProcessorCount,
                               0));
  CHECK(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, 0));
  CHECK(cudaDeviceGetAttribute(&subgroup_width, cudaDevAttrWarpSize, 0));
  CHECK(cudaDeviceGetAttribute(&compute_major,
                               cudaDevAttrComputeCapabilityMajor, 0));
  // BF16 WMMA needs sm_80. The ptx version reports the architecture this
  // image was compiled for.
  cudaFuncAttributes wmma_attr;
  CHECK(cudaFuncGetAttributes(&wmma_attr, fwd_wmma_kernel));
  const bool mma_supported = compute_major >= 8 && wmma_attr.ptxVersion >= 80;
  const SSM2DeviceInfo device = {(size_t)compute_units, max_threads,
                                 subgroup_width};
  printf("Launch policy is heuristic; benchmark tile and thread choices for "
         "each target GPU.\n");
  size_t fwd_cap = 0, out_cap = 0, state_cap = 0, tiled_cap = 0;
  size_t mma_state_cap = 0, mma_out_cap = 0;
  bool ok = check;
  int ran = 0;
  printf("\nchunk  chunks  path          ms  check\n");
  for (int chunk : kSweepChunks) {
    if (T % chunk != 0) {
      printf("%5d  skipped (tokens %% chunk != 0)\n", chunk);
      continue;
    }
    ChunkPlan plan;
    if (!plan_chunk(B, T, H, G, P, N, chunk, max_smem, mma_supported, device,
                    &plan)) {
      fprintf(stderr, "chunk %d allocation size overflows size_t\n", chunk);
      ok = false;
      continue;
    }
    p.chunk = chunk;
    p.nchunks = T / chunk;

    // Workspaces depend on nchunks and on whether output reuses a materialized
    // C @ B^T matrix. They are released after timing this chunk size.
    float *d_cb = nullptr;
    ssm2_bf16 *d_hiddens = nullptr;
    if (plan.cb_elems)
      CHECK(cudaMalloc(&d_cb, plan.cb_elems * sizeof(float)));
    CHECK(cudaMalloc(&d_hiddens, plan.hid_elems * sizeof(ssm2_bf16)));
    p.cb = d_cb;
    p.hiddens = d_hiddens;

    // Dynamic shared-memory requests above CUDA's default per-block limit must
    // be explicitly opted in once for each kernel.
    if (plan.use_mma) {
      if (plan.state_smem > mma_state_cap) {
        CHECK(cudaFuncSetAttribute(fwd_wmma_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)plan.state_smem));
        mma_state_cap = plan.state_smem;
      }
      if (plan.out_smem > mma_out_cap) {
        CHECK(cudaFuncSetAttribute(out_wmma_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)plan.out_smem));
        mma_out_cap = plan.out_smem;
      }
    } else if (plan.use_gemm) {
      if (plan.fwd_smem > fwd_cap) {
        CHECK(cudaFuncSetAttribute(fwd_gemm_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)plan.fwd_smem));
        fwd_cap = plan.fwd_smem;
      }
      if (plan.out_smem > out_cap) {
        CHECK(cudaFuncSetAttribute(compute_out_gemm_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)plan.out_smem));
        out_cap = plan.out_smem;
      }
    } else {
      if (plan.smem_state && plan.state_smem > state_cap) {
        CHECK(cudaFuncSetAttribute(fwd_pass_state_tiled_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)plan.state_smem));
        state_cap = plan.state_smem;
      }
      if (plan.smem_out && plan.out_smem > tiled_cap) {
        CHECK(cudaFuncSetAttribute(compute_out_tiled_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)plan.out_smem));
        tiled_cap = plan.out_smem;
      }
    }

    // Validate one untimed forward pass before warmup can overwrite its result.
    launch_forward(p, plan.use_mma, plan.use_gemm, plan.smem_state, plan.smem_out,
                   plan.state_bh, plan.state_bs, plan.out_bh, plan.state_splits,
                   max_grid_x, plan.state_smem, plan.fwd_smem, plan.cb_smem,
                   plan.out_smem);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out16, d_out, x_elems * sizeof(ssm2_bf16),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_state, d_state, st_elems * sizeof(float),
                     cudaMemcpyDeviceToHost));
    const char *check_text = "skipped";
    if (check) {
      for (size_t i = 0; i < x_elems; ++i)
        h_out[i] = ssm2_from_bf16(h_out16[i]);
      const bool chunk_ok =
          all_close(h_out, h_out_ref, x_elems, kOutTol) &&
          all_close(h_state, h_state_ref, st_elems, kStateTol);
      ok = ok && chunk_ok;
      check_text = chunk_ok ? "PASS" : "FAIL";
    }

    // Warmups stabilize clocks and caches. Default-stream ordering plus the
    // final synchronization makes the measured interval cover complete passes.
    const int warmup = 100;
    for (int i = 0; i < warmup; ++i)
      launch_forward(p, plan.use_mma, plan.use_gemm, plan.smem_state,
                     plan.smem_out,
                     plan.state_bh, plan.state_bs, plan.out_bh,
                     plan.state_splits, max_grid_x, plan.state_smem,
                     plan.fwd_smem, plan.cb_smem, plan.out_smem);
    CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i)
      launch_forward(p, plan.use_mma, plan.use_gemm, plan.smem_state,
                     plan.smem_out,
                     plan.state_bh, plan.state_bs, plan.out_bh,
                     plan.state_splits, max_grid_x, plan.state_smem,
                     plan.fwd_smem, plan.cb_smem, plan.out_smem);
    CHECK(cudaDeviceSynchronize());
    auto stop = std::chrono::steady_clock::now();
    CHECK(cudaGetLastError());
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count() /
        repeat;
    printf("%5d  %6d  %-12s  %8.3f  %s\n", chunk, p.nchunks,
           plan.use_mma ? "wmma" : (plan.use_gemm ? "gemm" : "scalar"), ms,
           check_text);
    ++ran;
    CHECK(cudaFree(d_cb));
    CHECK(cudaFree(d_hiddens));
  }
  if (!ran) {
    fprintf(stderr, "No sweep chunk divides tokens (%d)\n", T);
    ok = false;
  }

  // Release all shape-independent host and device allocations.
  free(h_x);
  free(h_dt);
  free(h_a);
  free(h_d);
  free(h_b);
  free(h_c);
  free(h_init);
  free(h_out);
  free(h_out_ref);
  free(h_state);
  free(h_state_ref);
  free(h_x16);
  free(h_dt16);
  free(h_b16);
  free(h_c16);
  free(h_out16);
  CHECK(cudaFree(d_x));
  CHECK(cudaFree(d_dt));
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_d));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));
  CHECK(cudaFree(d_init));
  CHECK(cudaFree(d_decay));
  CHECK(cudaFree(d_state));
  CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}
