// Fused per-head QK RMSNorm and rotate-half MRoPE from the attention block of
// Ideogram v4 (https://github.com/ideogram-oss/ideogram4):
//
//   y = bf16(x * rsqrt(mean(x^2) + eps) * w)
//   rotate-half RoPE on all of y, then store as bf16
//
// Q and K are normalized separately and share one position. V is not part of
// this kernel. The model head size is 256 (4608 / 18). RoPE covers the whole
// head. eps is 1e-5, the QK norm epsilon.
//
// Layout, row-major bf16 except the position and axis tables:
//   q, k          [tokens, heads, head_dim]
//   q_out, k_out  [tokens, heads, head_dim]
//   q_weight, k_weight                     [head_dim], shared by every head
//   cos_sin_cache [max_pos, head_dim]      first half cos, second half sin
//   positions     [3, tokens]              rows T, H, W
//   axis_map      [head_dim / 2]           which of the 3 rows owns a lane
//
// Sections (t, h, w) come from the command line. Ideogram v4 uses (24, 20, 20).
// Lanes start on T. H overwrites 1, 4, 7, ... below 3*h, and W overwrites
// 2, 5, 8, ... below 3*w. The temporal section is the remainder inside the
// first t+h+w lanes; this follows Ideogram4MRoPE, which does not walk t as
// its own span. On a head of 256, (24, 20, 20) leaves 88 temporal, 20 height,
// and 20 width lanes. A smaller head clamps each span to head_dim / 2.

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(__HIP__)
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
using bf16 = __hip_bfloat16;
using gpuError_t = hipError_t;
#define gpuGetErrorString hipGetErrorString
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemset hipMemset
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetLastError hipGetLastError
#define gpuSuccess hipSuccess
// HIP shuffle width is the subgroup size. It must stay inside one wavefront.
#define shfl_xor(v, m, w) __shfl_xor(v, m, w)
#define shfl_idx(v, s, w) __shfl(v, s, w)
#else
#include <cuda_bf16.h>
#include <cuda_runtime.h>
using bf16 = __nv_bfloat16;
using gpuError_t = cudaError_t;
#define gpuGetErrorString cudaGetErrorString
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemset cudaMemset
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuSuccess cudaSuccess
// The mask covers one 32-lane CUDA warp. Callers only use widths 8, 16, and 32.
#define shfl_xor(v, m, w) __shfl_xor_sync(0xffffffffu, v, m, w)
#define shfl_idx(v, s, w) __shfl_sync(0xffffffffu, v, s, w)
#endif

static inline float bf16_to_float(bf16 x) { return __bfloat162float(x); }
static inline bf16 float_to_bf16(float x) { return __float2bfloat16(x); }

#include "reference.h"

static void gpu_check(gpuError_t error, const char* file, int line) {
  if (error != gpuSuccess) {
    printf("[GPU ERROR] at file %s:%d:\n%s\n", file, line, gpuGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}
#define gpuCheck(err) gpu_check(err, __FILE__, __LINE__)

// 3 * section, clamped to half. section >= (half + 2) / 3 means the product
// reaches or passes half, including when 3 * section would overflow int.
static int interleaved_limit(int section, int half) {
  if (section >= (half + 2) / 3)
    return half;
  return 3 * section;
}

// axis[lane] is 0, 1, or 2. Height and width sections set the overwrite spans.
// The temporal section is not a span; see the file comment.
static void build_axis_map(int half, int section_h, int section_w, int* axis) {
  for (int lane = 0; lane < half; ++lane)
    axis[lane] = 0;
  const int limit_h = interleaved_limit(section_h, half);
  const int limit_w = interleaved_limit(section_w, half);
  for (int lane = 1; lane < limit_h; lane += 3)
    axis[lane] = 1;
  for (int lane = 2; lane < limit_w; lane += 3)
    axis[lane] = 2;
}

static void count_axes(const int* axis, int half, int counts[3]) {
  counts[0] = counts[1] = counts[2] = 0;
  for (int lane = 0; lane < half; ++lane)
    ++counts[axis[lane]];
}

// A uint4 load of bf16 is eight elements, stored as four little-endian pairs.
// Shifting the low 16 bits into the float exponent field recovers the value
// without a convert intrinsic. Packing rounds each float to bf16.
__device__ __forceinline__ float bf16_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(a))) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(b))) << 16);
}

__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
  f[0] = bf16_lo(v.x); f[1] = bf16_hi(v.x);
  f[2] = bf16_lo(v.y); f[3] = bf16_hi(v.y);
  f[4] = bf16_lo(v.z); f[5] = bf16_hi(v.z);
  f[6] = bf16_lo(v.w); f[7] = bf16_hi(v.w);
}

// A block has BLOCK lanes and covers BLOCK / (head_dim / 8) consecutive heads,
// which may span tokens. Each token contributes its Q heads followed by its K
// heads. Lane g of a head owns elements [8g, 8g + 8) and loads them as one uint4.
//
// Rotate-half pairs element i with element i + head_dim/2. Those two live on
// lanes g and g + (head_dim/2)/8, so the partner is one indexed shuffle inside
// the head's lane group.
//
// Sum of squares is reduced with xor shuffles across that same group, so the
// kernel takes no block barrier. Lanes past the last head still run the
// shuffles: they read head 0 and skip the stores. Dropping them would make
// the shuffle source undefined for the last partial group.
//
// Interleaved MRoPE reads the T, H, and W cos/sin rows for this token, then
// selects one row per element with axis_map. The rows are loaded before the
// reduction because they do not depend on x.
template <int HEAD_DIM, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
qk_rmsnorm_mrope(const bf16* __restrict__ q,
                 const bf16* __restrict__ k,
                 bf16* __restrict__ q_out,
                 bf16* __restrict__ k_out,
                 const bf16* __restrict__ q_weight,
                 const bf16* __restrict__ k_weight,
                 const bf16* __restrict__ cos_sin_cache,
                 const int64_t* __restrict__ positions,
                 const int* __restrict__ axis_map,
                 int tokens, int heads, int pos_stride, float eps) {
  constexpr int VEC = 8;
  constexpr int G = HEAD_DIM / VEC;
  constexpr int HEADS_PER_BLOCK = BLOCK / G;
  constexpr int HALF = HEAD_DIM / 2;
  static_assert(BLOCK % G == 0, "a block must hold whole heads");

  const int g = threadIdx.x % G;
  const int nheads = heads * 2;
  const int total = tokens * nheads;
  const int64_t gh = static_cast<int64_t>(blockIdx.x) * HEADS_PER_BLOCK + threadIdx.x / G;
  const bool valid = gh < total;
  const int ghc = valid ? static_cast<int>(gh) : 0;
  const int token = ghc / nheads;
  const int head = ghc % nheads;
  const bool is_k = head >= heads;
  const int local = is_k ? head - heads : head;
  const int d0 = g * VEC;

  const bf16* in_base;
  const bf16* weight;
  bf16* out_base;
  if (is_k) {
    in_base = k + (static_cast<int64_t>(token) * heads + local) * HEAD_DIM;
    weight = k_weight;
    out_base = k_out + (static_cast<int64_t>(token) * heads + local) * HEAD_DIM;
  } else {
    in_base = q + (static_cast<int64_t>(token) * heads + local) * HEAD_DIM;
    weight = q_weight;
    out_base = q_out + (static_cast<int64_t>(token) * heads + local) * HEAD_DIM;
  }

  const uint4 xv = *reinterpret_cast<const uint4*>(in_base + d0);
  const uint4 wv = *reinterpret_cast<const uint4*>(weight + d0);

  const int h8 = HALF / VEC;
  const bool first = g < h8;
  const int c0 = first ? d0 : d0 - HALF;

  constexpr int ROWS = 3;
  uint4 cv[ROWS], sv[ROWS];
#pragma unroll
  for (int a = 0; a < ROWS; ++a) {
    const int64_t pos = positions[static_cast<int64_t>(a) * pos_stride + token];
    const bf16* row = cos_sin_cache + pos * HEAD_DIM;
    cv[a] = *reinterpret_cast<const uint4*>(row + c0);
    sv[a] = *reinterpret_cast<const uint4*>(row + HALF + c0);
  }

  float x[VEC], w[VEC];
  unpack8(xv, x);
  unpack8(wv, w);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < VEC; ++i)
    ss += x[i] * x[i];
#pragma unroll
  for (int m = G / 2; m > 0; m >>= 1)
    ss += shfl_xor(ss, m, G);
  const float inv_rms = rsqrtf(ss / static_cast<float>(HEAD_DIM) + eps);

  uint32_t n[VEC / 2];
#pragma unroll
  for (int i = 0; i < VEC / 2; ++i)
    n[i] = pack_bf16x2(x[2 * i] * inv_rms * w[2 * i],
                       x[2 * i + 1] * inv_rms * w[2 * i + 1]);

  const int src = first ? g + h8 : g - h8;
  uint32_t p[VEC / 2];
#pragma unroll
  for (int i = 0; i < VEC / 2; ++i)
    p[i] = shfl_idx(n[i], src, G);

  float c[VEC], s[VEC];
  const int4 a_lo = *reinterpret_cast<const int4*>(axis_map + c0);
  const int4 a_hi = *reinterpret_cast<const int4*>(axis_map + c0 + 4);
  const int axes[VEC] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w, a_hi.x, a_hi.y, a_hi.z, a_hi.w};
  float cr[3][VEC], sr[3][VEC];
#pragma unroll
  for (int a = 0; a < 3; ++a) {
    unpack8(cv[a], cr[a]);
    unpack8(sv[a], sr[a]);
  }
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    c[i] = axes[i] == 0 ? cr[0][i] : (axes[i] == 1 ? cr[1][i] : cr[2][i]);
    s[i] = axes[i] == 0 ? sr[0][i] : (axes[i] == 1 ? sr[1][i] : sr[2][i]);
  }

  uint32_t r[VEC / 2];
#pragma unroll
  for (int i = 0; i < VEC / 2; ++i) {
    const float me0 = bf16_lo(n[i]), me1 = bf16_hi(n[i]);
    const float pa0 = bf16_lo(p[i]), pa1 = bf16_hi(p[i]);
    float o0, o1;
    if (first) {
      o0 = me0 * c[2 * i] - pa0 * s[2 * i];
      o1 = me1 * c[2 * i + 1] - pa1 * s[2 * i + 1];
    } else {
      o0 = me0 * c[2 * i] + pa0 * s[2 * i];
      o1 = me1 * c[2 * i + 1] + pa1 * s[2 * i + 1];
    }
    r[i] = pack_bf16x2(o0, o1);
  }

  if (valid)
    *reinterpret_cast<uint4*>(out_base + d0) = make_uint4(r[0], r[1], r[2], r[3]);
}

template <int HEAD_DIM, int BLOCK>
static void launch_block(const bf16* q, const bf16* k, bf16* q_out, bf16* k_out,
                         const bf16* q_weight, const bf16* k_weight, const bf16* cache,
                         const int64_t* positions, const int* axis_map, int tokens, int heads,
                         int pos_stride, float eps) {
  constexpr int heads_per_block = BLOCK / (HEAD_DIM / 8);
  const int64_t total = static_cast<int64_t>(tokens) * heads * 2;
  const unsigned blocks = static_cast<unsigned>((total + heads_per_block - 1) / heads_per_block);
  qk_rmsnorm_mrope<HEAD_DIM, BLOCK><<<blocks, BLOCK>>>(
      q, k, q_out, k_out, q_weight, k_weight, cache, positions, axis_map, tokens, heads,
      pos_stride, eps);
}

static const int block_sizes[] = {64, 128, 256, 512, 1024};

template <int HEAD_DIM>
static void launch(const bf16* q, const bf16* k, bf16* q_out, bf16* k_out,
                   const bf16* q_weight, const bf16* k_weight, const bf16* cache,
                   const int64_t* positions, const int* axis_map, int tokens, int heads,
                   int pos_stride, float eps, int block_size) {
  switch (block_size) {
    case 64:
      launch_block<HEAD_DIM, 64>(q, k, q_out, k_out, q_weight, k_weight, cache, positions,
                                 axis_map, tokens, heads, pos_stride, eps);
      break;
    case 128:
      launch_block<HEAD_DIM, 128>(q, k, q_out, k_out, q_weight, k_weight, cache, positions,
                                  axis_map, tokens, heads, pos_stride, eps);
      break;
    case 256:
      launch_block<HEAD_DIM, 256>(q, k, q_out, k_out, q_weight, k_weight, cache, positions,
                                  axis_map, tokens, heads, pos_stride, eps);
      break;
    case 512:
      launch_block<HEAD_DIM, 512>(q, k, q_out, k_out, q_weight, k_weight, cache, positions,
                                  axis_map, tokens, heads, pos_stride, eps);
      break;
    case 1024:
      launch_block<HEAD_DIM, 1024>(q, k, q_out, k_out, q_weight, k_weight, cache, positions,
                                   axis_map, tokens, heads, pos_stride, eps);
      break;
    default:
      printf("unsupported block size %d\n", block_size);
      exit(EXIT_FAILURE);
  }
}

// Device rsqrtf and the host libm sqrt do not match bit for bit, and the two
// reductions add the squares in a different order. 2e-2 covers that after the
// bf16 round.
static bool close_enough(float got, float ref) {
  const float atol = 2e-2f;
  const float rtol = 2e-2f;
  return fabsf(got - ref) <= atol + rtol * fabsf(ref);
}

static int run(int tokens, int heads, int head_dim, int section_t, int section_h, int section_w,
               int repeat) {
  int status = 0;

  if (tokens <= 0 || heads <= 0 || repeat <= 0) {
    printf("tokens, heads, and repeat must be positive\n");
    return 1;
  }
  if (static_cast<int64_t>(tokens) * heads * 2 > INT_MAX) {
    printf("tokens * heads * 2 must fit in int\n");
    return 1;
  }
  if (head_dim != 64 && head_dim != 128 && head_dim != 256) {
    printf("head_dim must be 64, 128, or 256\n");
    return 1;
  }
  if (section_t < 0 || section_h < 0 || section_w < 0) {
    printf("sections must be non-negative\n");
    return 1;
  }

  const int half = head_dim / 2;
  const float eps = 1e-5f;
  const int max_pos = 512;

  std::vector<int> axis_map(half, 0);
  int axis_counts[3] = {0, 0, 0};
  build_axis_map(half, section_h, section_w, axis_map.data());
  count_axes(axis_map.data(), half, axis_counts);
  if (axis_counts[0] + axis_counts[1] + axis_counts[2] != half) {
    printf("axis map does not cover %d lanes\n", half);
    return 1;
  }
  // t is the remainder of the first t+h+w lanes. When that prefix fits, the
  // lane counts have to equal the requested sections.
  const int span = section_t + section_h + section_w;
  if (span <= half) {
    int prefix[3] = {0, 0, 0};
    for (int lane = 0; lane < span; ++lane)
      ++prefix[axis_map[lane]];
    if (prefix[0] != section_t || prefix[1] != section_h || prefix[2] != section_w) {
      printf("sections %d %d %d do not match the first %d lanes (%d %d %d)\n", section_t,
             section_h, section_w, span, prefix[0], prefix[1], prefix[2]);
      return 1;
    }
  }

  const size_t qk_n = static_cast<size_t>(tokens) * heads * head_dim;
  const size_t pos_n = static_cast<size_t>(3) * tokens;
  const size_t cache_n = static_cast<size_t>(max_pos) * head_dim;

  std::vector<bf16> h_q(qk_n), h_k(qk_n), h_qw(head_dim), h_kw(head_dim), h_cache(cache_n);
  std::vector<bf16> h_qo(qk_n), h_ko(qk_n);
  std::vector<int64_t> h_pos(pos_n);

  srand(123);
  auto rnd = []() { return rand() / static_cast<float>(RAND_MAX) * 2.f - 1.f; };
  for (size_t i = 0; i < qk_n; ++i) {
    h_q[i] = float_to_bf16(rnd());
    h_k[i] = float_to_bf16(rnd());
  }
  for (int i = 0; i < head_dim; ++i) {
    h_qw[i] = float_to_bf16(0.5f + 0.5f * rnd());
    h_kw[i] = float_to_bf16(0.5f + 0.5f * rnd());
  }
  for (int p = 0; p < max_pos; ++p) {
    for (int lane = 0; lane < half; ++lane) {
      const float angle = (p + 1) * (lane + 1) * 0.01f;
      h_cache[p * head_dim + lane] = float_to_bf16(cosf(angle));
      h_cache[p * head_dim + half + lane] = float_to_bf16(sinf(angle));
    }
  }
  for (int64_t t = 0; t < tokens; ++t) {
    h_pos[t] = (t * 3 + 1) % max_pos;
    h_pos[tokens + t] = (t * 5 + 4) % max_pos;
    h_pos[2 * tokens + t] = (t * 7 + 9) % max_pos;
  }

  bf16 *d_q, *d_k, *d_qw, *d_kw, *d_cache, *d_qo, *d_ko;
  int64_t* d_pos;
  int* d_axis;
  gpuCheck(gpuMalloc(&d_q, qk_n * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_k, qk_n * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_qw, head_dim * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_kw, head_dim * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_cache, cache_n * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_qo, qk_n * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_ko, qk_n * sizeof(bf16)));
  gpuCheck(gpuMalloc(&d_pos, pos_n * sizeof(int64_t)));
  gpuCheck(gpuMalloc(&d_axis, half * sizeof(int)));
  gpuCheck(gpuMemcpy(d_q, h_q.data(), qk_n * sizeof(bf16), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_k, h_k.data(), qk_n * sizeof(bf16), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_qw, h_qw.data(), head_dim * sizeof(bf16), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_kw, h_kw.data(), head_dim * sizeof(bf16), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_cache, h_cache.data(), cache_n * sizeof(bf16), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_pos, h_pos.data(), pos_n * sizeof(int64_t), gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_axis, axis_map.data(), half * sizeof(int), gpuMemcpyHostToDevice));

  const int pos_stride = tokens;
  auto launch_kernel = [&](int block_size) {
    if (head_dim == 64)
      launch<64>(d_q, d_k, d_qo, d_ko, d_qw, d_kw, d_cache, d_pos, d_axis, tokens, heads,
                 pos_stride, eps, block_size);
    else if (head_dim == 128)
      launch<128>(d_q, d_k, d_qo, d_ko, d_qw, d_kw, d_cache, d_pos, d_axis, tokens, heads,
                  pos_stride, eps, block_size);
    else
      launch<256>(d_q, d_k, d_qo, d_ko, d_qw, d_kw, d_cache, d_pos, d_axis, tokens, heads,
                  pos_stride, eps, block_size);
  };

  // Compares h_qo and h_ko with the host reference and returns the number of
  // mismatches, stopping at 8.
  std::vector<float> ref(head_dim);
  auto count_faults = [&](int block_size) {
    int nfaults = 0;
    for (int t = 0; t < tokens && nfaults < 8; ++t) {
      for (int h = 0; h < heads && nfaults < 8; ++h) {
        const bf16* in = h_q.data() + (static_cast<int64_t>(t) * heads + h) * head_dim;
        qk_rmsnorm_rope_host(ref.data(), in, h_qw.data(), head_dim, h_cache.data(),
                             h_pos.data(), axis_map.data(), pos_stride, t, eps);
        const bf16* got = h_qo.data() + (static_cast<int64_t>(t) * heads + h) * head_dim;
        for (int i = 0; i < head_dim; ++i) {
          const float gv = bf16_to_float(got[i]);
          if (!close_enough(gv, ref[i])) {
            printf("block_size %d: mismatch of q at token %d head %d dim %d: CPU_ref %f vs GPU %f\n",
                   block_size, t, h, i, ref[i], gv);
            if (++nfaults >= 8)
              break;
          }
        }
      }
      for (int h = 0; h < heads && nfaults < 8; ++h) {
        const bf16* in = h_k.data() + (static_cast<int64_t>(t) * heads + h) * head_dim;
        qk_rmsnorm_rope_host(ref.data(), in, h_kw.data(), head_dim, h_cache.data(),
                             h_pos.data(), axis_map.data(), pos_stride, t, eps);
        const bf16* got = h_ko.data() + (static_cast<int64_t>(t) * heads + h) * head_dim;
        for (int i = 0; i < head_dim; ++i) {
          const float gv = bf16_to_float(got[i]);
          if (!close_enough(gv, ref[i])) {
            printf("block_size %d: mismatch of k at token %d head %d dim %d: CPU_ref %f vs GPU %f\n",
                   block_size, t, h, i, ref[i], gv);
            if (++nfaults >= 8)
              break;
          }
        }
      }
    }
    return nfaults;
  };

  // Check the kernel at every block size. The outputs are cleared first so a
  // result left by an earlier block size cannot pass.
  int nfaults = 0;
  for (int block_size : block_sizes) {
    gpuCheck(gpuMemset(d_qo, 0, qk_n * sizeof(bf16)));
    gpuCheck(gpuMemset(d_ko, 0, qk_n * sizeof(bf16)));
    launch_kernel(block_size);
    gpuCheck(gpuGetLastError());
    gpuCheck(gpuMemcpy(h_qo.data(), d_qo, qk_n * sizeof(bf16), gpuMemcpyDeviceToHost));
    gpuCheck(gpuMemcpy(h_ko.data(), d_ko, qk_n * sizeof(bf16), gpuMemcpyDeviceToHost));
    nfaults = count_faults(block_size);
    if (nfaults)
      break;
  }

  if (nfaults) {
    status = 1;
    printf("FAIL\n");
  } else {
    printf("PASS\n");
    printf("tokens %d heads %d head_dim %d sections %d %d %d axes %d %d %d\n", tokens, heads,
           head_dim, section_t, section_h, section_w, axis_counts[0], axis_counts[1],
           axis_counts[2]);

    // Bytes that must move once per launch: Q and K in, then Q and K out.
    // Weights are one head vector reused by every head, and each cos/sin row is
    // reused by every head of the token, so neither is counted.
    const size_t bytes = 4 * qk_n * sizeof(bf16);

    for (int block_size : block_sizes) {
      for (int i = 0; i < 100; ++i)
        launch_kernel(block_size);
      gpuCheck(gpuDeviceSynchronize());

      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; ++i)
        launch_kernel(block_size);
      gpuCheck(gpuDeviceSynchronize());
      auto stop = std::chrono::steady_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(stop - start).count() / repeat;
      const double gbps = (bytes / ms) / 1e6;
      printf("block_size %4d | Average kernel execution time: %f (ms) | bandwidth %.2f GB/s\n",
             block_size, ms, gbps);
    }
  }

  gpuCheck(gpuFree(d_q));
  gpuCheck(gpuFree(d_k));
  gpuCheck(gpuFree(d_qw));
  gpuCheck(gpuFree(d_kw));
  gpuCheck(gpuFree(d_cache));
  gpuCheck(gpuFree(d_qo));
  gpuCheck(gpuFree(d_ko));
  gpuCheck(gpuFree(d_pos));
  gpuCheck(gpuFree(d_axis));

  return status;
}

int main(int argc, char** argv) {
  if (argc != 8) {
    printf("Usage: %s <tokens> <heads> <head_dim> <section_t> <section_h> <section_w> <repeat>\n",
           argv[0]);
    printf("  head_dim 64, 128, or 256 (Ideogram v4 uses 256)\n");
    printf("  sections are interleaved MRoPE (t, h, w); Ideogram v4 uses 24 20 20\n");
    return 1;
  }
  return run(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
             atoi(argv[6]), atoi(argv[7]));
}
