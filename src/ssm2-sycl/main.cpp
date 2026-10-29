// Mamba-2 SSD forward pass, SYCL.
//
//   chunk_cumsum, then joint_matrix on aligned 16-wide tiles, the
//   register-blocked GEMM path, or the scalar
//   tiled/global state and output paths. Work-groups grid-stride when a
//   logical block count exceeds the device's maximum work-group count.
//
// Data types follow TileGym's mamba2_chunk_forward; see ssm2_types.h.


#include <chrono>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
#include <new>
#include <random>
#include <string>
#include <sycl/sycl.hpp>

#include "ssm2_launch_policy.h"
#include "kernels.h"

constexpr int kSweepChunks[] = {32, 64, 128};

// Element counts use size_t. Extents themselves stay in int because each one
// is a work-group or loop bound; only the flattened offset can exceed 2^31.
static bool mul_dims(size_t *out, std::initializer_list<int> dims) {
  size_t value = 1;
  for (int dim : dims) {
    if (dim <= 0)
      return false;
    const size_t d = (size_t)dim;
    if (value > SIZE_MAX / d)
      return false;
    value *= d;
  }
  *out = value;
  return true;
}

// Convert a validated element count to bytes without overflowing size_t.
static bool float_elems(size_t *out, std::initializer_list<int> dims) {
  size_t value = 0;
  if (!mul_dims(&value, dims) || value > SIZE_MAX / sizeof(float))
    return false;
  *out = value;
  return true;
}

static bool parse_positive(const char *text, const char *name, int *out) {
  char *end = nullptr;
  const long value = strtol(text, &end, 10);
  if (end == text || *end || value <= 0 || value > 1000000000L) {
    fprintf(stderr, "Invalid %s: %s\n", name, text);
    return false;
  }
  *out = (int)value;
  return true;
}

// TileGym's forward test tolerances (atol = rtol) against its FP32 reference.
constexpr float kOutTol = 2e-2f;
constexpr float kStateTol = 1e-2f;

// Apply torch.testing.assert_close's |got - ref| <= atol + rtol * |ref| test
// with atol = rtol = tol; non-finite values fail.
static bool all_close(const float *got, const float *ref, size_t n, float tol) {
  size_t bad = 0;
#pragma omp parallel for schedule(static) reduction(+ : bad)
  for (long long i = 0; i < (long long)n; ++i) {
    const bool close =
        isfinite(got[i]) && isfinite(ref[i]) &&
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

// icpx and clang++ share this query. The matrix shape follows the GPU, not
// which of those compilers produced the binary.
static bool vendor_has(const std::string &text, const char *lower) {
  const size_t n = std::char_traits<char>::length(lower);
  if (n == 0 || text.size() < n)
    return false;
  for (size_t i = 0; i + n <= text.size(); ++i) {
    size_t j = 0;
    for (; j < n; ++j) {
      unsigned char c = static_cast<unsigned char>(text[i + j]);
      if (c >= 'A' && c <= 'Z')
        c = static_cast<unsigned char>(c - 'A' + 'a');
      if (c != static_cast<unsigned char>(lower[j]))
        break;
    }
    if (j == n)
      return true;
  }
  return false;
}

enum class Ssm2Gpu { Intel, NvidiaAmd, Other };

static Ssm2Gpu ssm2_gpu(const sycl::device &dev) {
  const std::string vendor = dev.get_info<sycl::info::device::vendor>();
  if (vendor_has(vendor, "intel"))
    return Ssm2Gpu::Intel;
  if (vendor_has(vendor, "nvidia") || vendor_has(vendor, "amd") ||
      vendor_has(vendor, "advanced micro"))
    return Ssm2Gpu::NvidiaAmd;
  return Ssm2Gpu::Other;
}

// Launch one complete SSD forward pass. Logical group counts use size_t and
// kernels group-stride when a count exceeds the device's work-group count.
static void launch_forward(sycl::queue &q, const Params &p, bool use_mma,
                           bool use_gemm, bool smem_state, bool smem_out,
                           int state_bh, int state_bs, int out_bh,
                           int state_splits, size_t max_groups,
                           int scalar_threads, size_t state_smem,
                           size_t fwd_smem, size_t cb_smem, size_t out_smem,
                           int mma_sg, bool intel) {
  launch_cumsum(q, p, max_groups);
  // Intel uses the 8x16x16 shape inside the GEMM path. NVIDIA and AMD use the
  // 16x16 joint_matrix when the subgroup is 32 or 64.
  if (!intel && use_mma && (mma_sg == 32 || mma_sg == 64)) {
    if (mma_sg == 64)
      launch_mma<64>(q, p, max_groups);
    else
      launch_mma<32>(q, p, max_groups);
    return;
  }
  if (use_gemm) {
    launch_gemm(q, p, use_mma, state_bh, state_bs, state_splits, out_bh,
                max_groups, fwd_smem, cb_smem, out_smem);
    return;
  }
  if (smem_state)
    launch_state_tiled(q, p, max_groups, state_smem);
  else
    launch_state_global(q, p, max_groups, scalar_threads);
  if (smem_out) {
    launch_cb(q, p, max_groups, scalar_threads);
    launch_out_tiled(q, p, max_groups, scalar_threads, out_smem);
  } else {
    launch_out_global(q, p, max_groups, scalar_threads);
  }
}

struct ChunkPlan {
  bool use_mma;
  bool use_gemm;
  bool smem_state;
  bool smem_out;
  int state_bh, state_bs, out_bh, state_splits;
  size_t state_smem, fwd_smem, cb_smem, out_smem, cb_elems, hid_elems;
};

// Validate a chunk size and derive its kernel, workspace, and local-memory
// settings. Thread and local-memory limits come from the SYCL device; the
// admission rules match the CUDA planner.
static bool plan_chunk(int B, int T, int H, int G, int P, int N, int chunk,
                       size_t max_smem, int max_threads, bool mma_supported,
                       bool intel, const SSM2DeviceInfo &device,
                       ChunkPlan *plan) {
  if (T % chunk != 0 || chunk > 1024 || chunk > max_threads)
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
  const int thread_cap = max_threads < 1024 ? max_threads : 1024;

  const int cb_threads = (chunk / 4) * (chunk / 4);
  const int out_threads = out_bh ? (chunk / 4) * (out_bh / 4) : 0;
  const int tile_threads =
      (state_bh && state_bs) ? (state_bh / 4) * (state_bs / 4) : 0;
  const int splits = launch.state_splits;
  const int state_threads = tile_threads * splits;

  // Local memory holds FP32 copies of the BF16 operands.
  const size_t fwd_smem =
      ((size_t)chunk * (state_bh + state_bs) + (size_t)state_threads * 16) *
      sizeof(float);
  const size_t cb_smem = (size_t)8 * chunk * sizeof(float);
  const size_t gemm_out_smem =
      (size_t)(chunk + chunk * chunk + chunk * out_bh + 4 * chunk +
               4 * out_bh) *
      sizeof(float);
  const bool use_gemm =
      chunk % 4 == 0 && chunk <= 128 && state_bh && state_bs && out_bh &&
      cb_threads > 0 && cb_threads <= thread_cap && out_threads > 0 &&
      out_threads <= thread_cap && state_threads > 0 &&
      state_threads <= thread_cap && fwd_smem <= max_smem &&
      gemm_out_smem <= max_smem;
  const size_t state_smem =
      (size_t)chunk * (kStatePTile + kStateNTile) * sizeof(float);
  const size_t tiled_out_smem =
      (size_t)(chunk + chunk * kOutPTile + chunk * chunk) * sizeof(float);
  // joint_matrix stages BF16 weighted B plus a 16x16 state tile, and FP32
  // dt * X plus a 16x16 accumulator and 16x8 TF32 operand tiles. Each 16x16
  // tile is one wavefront-sized work-group (32 or 64).
  const size_t mma_state_smem =
      (size_t)chunk * 16 * sizeof(ssm2_bf16) + 16 * 16 * sizeof(float);
  const size_t mma_out_smem =
      ((size_t)chunk * 16 + 16 * 16 + 16 * 8) * sizeof(float);
  if (intel) {
    // The 8x16x16 joint_matrix replaces only the GEMM path's C @ B^T.
    (void)mma_state_smem, (void)mma_out_smem;
    plan->use_mma = mma_supported && use_gemm && chunk % 16 == 0 && N % 16 == 0;
    plan->use_gemm = use_gemm;
  } else {
    plan->use_mma = mma_supported && chunk % 16 == 0 && P % 16 == 0 &&
                    N % 16 == 0 && kThreads <= max_threads &&
                    mma_state_smem <= max_smem && mma_out_smem <= max_smem;
    plan->use_gemm = use_gemm && !plan->use_mma;
  }
  // The tiled state kernel assigns one 16x16 element to each of 256 work-items.
  plan->smem_state =
      !plan->use_mma && !plan->use_gemm && state_smem <= max_smem &&
      kThreads <= max_threads;
  plan->smem_out =
      !plan->use_mma && !plan->use_gemm && tiled_out_smem <= max_smem;
  plan->state_bh = state_bh;
  plan->state_bs = state_bs;
  plan->out_bh = out_bh;
  plan->state_splits = splits;
  plan->state_smem = state_smem;
  plan->fwd_smem = fwd_smem;
  plan->cb_smem = cb_smem;
  plan->out_smem = use_gemm ? gemm_out_smem : tiled_out_smem;
  plan->cb_elems =
      (plan->use_mma || plan->use_gemm || plan->smem_out) ? cb_elems : 0;
  plan->hid_elems = hid_elems;
  return true;
}

static void free_host(float *h_x, float *h_dt, float *h_a, float *h_d,
                       float *h_b, float *h_c, float *h_init, float *h_out,
                       float *h_out_ref, float *h_state, float *h_state_ref,
                       ssm2_bf16 *h_x16, ssm2_bf16 *h_dt16, ssm2_bf16 *h_b16,
                       ssm2_bf16 *h_c16, ssm2_bf16 *h_out16) {
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
}

template <class T> static T *device_alloc(sycl::queue &q, size_t n) {
  T *ptr = sycl::malloc_device<T>(n, q);
  if (!ptr)
    throw std::bad_alloc();
  return ptr;
}

// Parse the benchmark shape, execute each chunk plan, and validate device results.
int main(int argc, char **argv) {
  float *h_x = nullptr, *h_dt = nullptr, *h_a = nullptr, *h_d = nullptr;
  float *h_b = nullptr, *h_c = nullptr, *h_init = nullptr, *h_out = nullptr;
  float *h_out_ref = nullptr, *h_state = nullptr, *h_state_ref = nullptr;
  ssm2_bf16 *h_x16 = nullptr, *h_dt16 = nullptr, *h_b16 = nullptr;
  ssm2_bf16 *h_c16 = nullptr, *h_out16 = nullptr;
  try {
  if (argc != 2 && argc != 8) {
    fprintf(
        stderr,
        "Usage: %s <repeat> [batch tokens heads groups head_size state_size]\n"
        "Defaults: repeat=100 B=2 T=256 H=8 G=2 P=32 N=64\n"
        "Sweeps chunk sizes 32, 64, and 128.\n",
        argv[0]);
    return EXIT_FAILURE;
  }

  int repeat = 100, B = 2, T = 256, H = 8, G = 2, P = 32, N = 64;
  if (!parse_positive(argv[1], "repeat", &repeat))
    return EXIT_FAILURE;
  if (argc == 8) {
    if (!parse_positive(argv[2], "batch", &B) ||
        !parse_positive(argv[3], "tokens", &T) ||
        !parse_positive(argv[4], "heads", &H) ||
        !parse_positive(argv[5], "groups", &G) ||
        !parse_positive(argv[6], "head_size", &P) ||
        !parse_positive(argv[7], "state_size", &N))
      return EXIT_FAILURE;
  }

  if (H % G != 0) {
    fprintf(stderr, "heads (%d) must be divisible by groups (%d)\n", H, G);
    return EXIT_FAILURE;
  }
  size_t x_elems = 0, dt_elems = 0, bc_elems = 0, st_elems = 0;
  if (!float_elems(&x_elems, {B, T, H, P}) ||
      !float_elems(&dt_elems, {B, T, H}) ||
      !float_elems(&bc_elems, {B, T, G, N}) ||
      !float_elems(&st_elems, {B, H, P, N})) {
    fprintf(stderr, "Problem size overflows size_t\n");
    return EXIT_FAILURE;
  }

  printf("ssm2 Mamba-2 SSD forward\n");
  printf("B=%d T=%d H=%d G=%d P=%d N=%d repeat=%d\n", B, T, H, G, P, N,
              repeat);
  printf("Chunk sweep:");
  for (int chunk : kSweepChunks)
    printf(" %d", chunk);
  printf("\n");

  // Host buffers hold deterministic inputs, the device result, and the scalar
  // oracle result. Element counts were checked before converting them to bytes.
  h_x = (float *)malloc(x_elems * sizeof(float));
  h_dt = (float *)malloc(dt_elems * sizeof(float));
  h_a = (float *)malloc((size_t)H * sizeof(float));
  h_d = (float *)malloc((size_t)H * sizeof(float));
  h_b = (float *)malloc(bc_elems * sizeof(float));
  h_c = (float *)malloc(bc_elems * sizeof(float));
  h_init = (float *)malloc(st_elems * sizeof(float));
  h_out = (float *)malloc(x_elems * sizeof(float));
  h_out_ref = (float *)malloc(x_elems * sizeof(float));
  h_state = (float *)malloc(st_elems * sizeof(float));
  h_state_ref = (float *)malloc(st_elems * sizeof(float));
  h_x16 = (ssm2_bf16 *)malloc(x_elems * sizeof(ssm2_bf16));
  h_dt16 = (ssm2_bf16 *)malloc(dt_elems * sizeof(ssm2_bf16));
  h_b16 = (ssm2_bf16 *)malloc(bc_elems * sizeof(ssm2_bf16));
  h_c16 = (ssm2_bf16 *)malloc(bc_elems * sizeof(ssm2_bf16));
  h_out16 = (ssm2_bf16 *)malloc(x_elems * sizeof(ssm2_bf16));
  if (!h_x || !h_dt || !h_a || !h_d || !h_b || !h_c || !h_init || !h_out ||
      !h_out_ref || !h_state || !h_state_ref || !h_x16 || !h_dt16 || !h_b16 ||
      !h_c16 || !h_out16) {
    fprintf(stderr, "Host allocation failed\n");
    free_host(h_x, h_dt, h_a, h_d, h_b, h_c, h_init, h_out, h_out_ref, h_state,
              h_state_ref, h_x16, h_dt16, h_b16, h_c16, h_out16);
    return EXIT_FAILURE;
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

  const int ref_threads = omp_get_max_threads();
  const size_t ref_limit =
      static_cast<size_t>(ref_threads < 1 ? 1 : ref_threads) * 2000000000ull;
  size_t ref_work = 0;
  const bool check = mul_dims(&ref_work, {B, T, H, P, N}) && ref_work <= ref_limit;
  if (check) {
    auto ref_start = std::chrono::steady_clock::now();
    ssm2_reference(h_x, h_a, h_b, h_c, h_d, h_dt, h_init, h_out_ref, h_state_ref,
                   B, T, H, G, P, N);
    auto ref_stop = std::chrono::steady_clock::now();
    printf("Reference (OpenMP, %d threads) time: %.3f ms\n", ref_threads,
                std::chrono::duration<double, std::milli>(ref_stop - ref_start)
                    .count());
  } else if (ref_work > ref_limit) {
    printf(
        "Reference skipped (B*T*H*P*N=%zu > %zu); validation will fail\n",
        ref_work, ref_limit);
  } else {
    printf("Reference skipped (B*T*H*P*N overflows size_t); validation "
                "will fail\n");
  }

  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
  const sycl::device sycl_device = q.get_device();

  const auto sg_sizes = sycl_device.get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int subgroup_width = *r;

  const size_t max_work_group =
      sycl_device.get_info<sycl::info::device::max_work_group_size>();
  const size_t max_item_size =
      sycl_device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0];
  // max_work_item_sizes bounds one work-group; the launch grid is bounded by
  // the work-group count, which core SYCL does not expose.
#ifdef SYCL_EXT_ONEAPI_MAX_WORK_GROUP_QUERY
  const size_t max_groups = sycl_device.get_info<
      sycl::ext::oneapi::experimental::info::device::max_work_groups<1>>()[0];
#else
  const size_t max_groups = (size_t)INT_MAX;
#endif
  const size_t max_local_mem =
      sycl_device.get_info<sycl::info::device::local_mem_size>();
  const size_t max_threads_sz =
      max_work_group < max_item_size ? max_work_group : max_item_size;
  const int max_threads =
      max_threads_sz > 2147483647u ? 2147483647 : (int)max_threads_sz;
  const int scalar_threads = max_threads < kThreads ? max_threads : kThreads;
  const SSM2DeviceInfo device = {
      (size_t)sycl_device.get_info<sycl::info::device::max_compute_units>(),
      max_threads, subgroup_width};
  // NVIDIA and AMD joint_matrix use the largest reported subgroup, 32 or 64.
  // Intel's 8x16x16 shape is a 16-lane subgroup, including on PVC where 32
  // is also reported. Both compilers see the same vendor and subgroup list.
  const Ssm2Gpu gpu = ssm2_gpu(sycl_device);
  const bool intel = gpu == Ssm2Gpu::Intel;
  int mma_sg = 0;
  if (intel) {
    for (auto sg : sg_sizes)
      if (sg == 16)
        mma_sg = 16;
  } else if (gpu == Ssm2Gpu::NvidiaAmd &&
             (subgroup_width == 32 || subgroup_width == 64)) {
    mma_sg = subgroup_width;
  }
  const bool mma_supported = mma_sg != 0;
  printf("Launch policy is heuristic; benchmark tile and thread choices "
              "for each target GPU.\n");

  ssm2_bf16 *d_x = device_alloc<ssm2_bf16>(q, x_elems);
  ssm2_bf16 *d_dt = device_alloc<ssm2_bf16>(q, dt_elems);
  float *d_a = device_alloc<float>(q, (size_t)H);
  float *d_d = device_alloc<float>(q, (size_t)H);
  ssm2_bf16 *d_b = device_alloc<ssm2_bf16>(q, bc_elems);
  ssm2_bf16 *d_c = device_alloc<ssm2_bf16>(q, bc_elems);
  float *d_init = device_alloc<float>(q, st_elems);
  float *d_decay = device_alloc<float>(q, dt_elems);
  float *d_state = device_alloc<float>(q, st_elems);
  ssm2_bf16 *d_out = device_alloc<ssm2_bf16>(q, x_elems);
  q.copy(h_x16, d_x, x_elems);
  q.copy(h_dt16, d_dt, dt_elems);
  q.copy(h_a, d_a, (size_t)H);
  q.copy(h_d, d_d, (size_t)H);
  q.copy(h_b16, d_b, bc_elems);
  q.copy(h_c16, d_c, bc_elems);
  q.copy(h_init, d_init, st_elems);

  Params p{B,   T,     H,      G,       P,       N,      0,
           0,   H / G, d_x,    d_a,     d_b,     d_c,    d_d,
           d_dt, d_init, d_decay, nullptr, nullptr, d_state, d_out};
  bool ok = check;
  int ran = 0;
  printf("\nchunk  chunks  path          ms  check\n");
  for (int chunk : kSweepChunks) {
    if (T % chunk != 0) {
      printf("%5d  skipped (tokens %% chunk != 0)\n", chunk);
      continue;
    }
    ChunkPlan plan;
    if (!plan_chunk(B, T, H, G, P, N, chunk, max_local_mem, max_threads,
                    mma_supported, intel, device, &plan)) {
      fprintf(stderr, "chunk %d allocation size overflows size_t\n",
                   chunk);
      ok = false;
      continue;
    }
    p.chunk = chunk;
    p.nchunks = T / chunk;

    float *d_cb = nullptr;
    ssm2_bf16 *d_hiddens = nullptr;
    if (plan.cb_elems)
      d_cb = device_alloc<float>(q, plan.cb_elems);
    d_hiddens = device_alloc<ssm2_bf16>(q, plan.hid_elems);
    p.cb = d_cb;
    p.hiddens = d_hiddens;

    launch_forward(q, p, plan.use_mma, plan.use_gemm,
                   plan.smem_state, plan.smem_out,
                   plan.state_bh, plan.state_bs, plan.out_bh, plan.state_splits,
                   max_groups, scalar_threads, plan.state_smem, plan.fwd_smem,
                   plan.cb_smem, plan.out_smem, mma_sg, intel);
    q.copy(d_out, h_out16, x_elems);
    q.copy(d_state, h_state, st_elems);
    q.wait_and_throw();
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

    const int warmup = 100;
    for (int i = 0; i < warmup; ++i)
      launch_forward(q, p, plan.use_mma, plan.use_gemm,
                     plan.smem_state, plan.smem_out,
                     plan.state_bh, plan.state_bs, plan.out_bh,
                     plan.state_splits, max_groups, scalar_threads,
                     plan.state_smem, plan.fwd_smem, plan.cb_smem,
                     plan.out_smem, mma_sg, intel);
    q.wait();

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i)
      launch_forward(q, p, plan.use_mma, plan.use_gemm,
                     plan.smem_state, plan.smem_out,
                     plan.state_bh, plan.state_bs, plan.out_bh,
                     plan.state_splits, max_groups, scalar_threads,
                     plan.state_smem, plan.fwd_smem, plan.cb_smem,
                     plan.out_smem, mma_sg, intel);
    q.wait();
    auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count() /
        repeat;
    printf("%5d  %6d  %-12s  %8.3f  %s\n", chunk, p.nchunks,
                plan.use_mma ? "joint_matrix"
                             : (plan.use_gemm ? "gemm" : "scalar"),
                ms, check_text);
    ++ran;
    sycl::free(d_cb, q);
    sycl::free(d_hiddens, q);
  }
  if (!ran) {
    fprintf(stderr, "No sweep chunk divides tokens (%d)\n", T);
    ok = false;
  }

  sycl::free(d_x, q);
  sycl::free(d_dt, q);
  sycl::free(d_a, q);
  sycl::free(d_d, q);
  sycl::free(d_b, q);
  sycl::free(d_c, q);
  sycl::free(d_init, q);
  sycl::free(d_decay, q);
  sycl::free(d_state, q);
  sycl::free(d_out, q);
  free_host(h_x, h_dt, h_a, h_d, h_b, h_c, h_init, h_out, h_out_ref, h_state,
            h_state_ref, h_x16, h_dt16, h_b16, h_c16, h_out16);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const sycl::exception &e) {
    fprintf(stderr, "SYCL error: %s\n", e.what());
  } catch (const std::bad_alloc &) {
    fprintf(stderr, "Device allocation failed\n");
  }
  free_host(h_x, h_dt, h_a, h_d, h_b, h_c, h_init, h_out, h_out_ref, h_state,
            h_state_ref, h_x16, h_dt16, h_b16, h_c16, h_out16);
  return EXIT_FAILURE;
}
