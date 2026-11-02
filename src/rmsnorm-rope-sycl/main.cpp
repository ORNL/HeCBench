// Fused per-head QK RMSNorm and rotate-half MRoPE
// (https://github.com/ideogram-oss/ideogram4).

#include <algorithm>
#include <chrono>
#include <type_traits>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sycl/sycl.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;

static inline float bf16_to_float(bf16 x) { return static_cast<float>(x); }
static inline bf16 float_to_bf16(float x) { return bf16(x); }

#include "reference.h"

// 3 * section, clamped to half. section >= (half + 2) / 3 means the product
// reaches or passes half, including when 3 * section would overflow int.
static int interleaved_limit(int section, int half) {
  if (section >= (half + 2) / 3)
    return half;
  return 3 * section;
}

// axis[lane] is 0, 1, or 2. Height and width sections set the overwrite spans.
// The temporal section is not a span; see the CUDA file comment.
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
// without a convert. Packing rounds each float to bf16.
static inline float bf16_lo(uint32_t u) { return sycl::bit_cast<float>(u << 16); }
static inline float bf16_hi(uint32_t u) { return sycl::bit_cast<float>(u & 0xffff0000u); }

static inline uint32_t pack_bf16x2(float a, float b) {
  return static_cast<uint32_t>(sycl::bit_cast<uint16_t>(bf16(a))) |
         (static_cast<uint32_t>(sycl::bit_cast<uint16_t>(bf16(b))) << 16);
}

static inline void unpack8(const sycl::uint4& v, float (&f)[8]) {
  f[0] = bf16_lo(v.x()); f[1] = bf16_hi(v.x());
  f[2] = bf16_lo(v.y()); f[3] = bf16_hi(v.y());
  f[4] = bf16_lo(v.z()); f[5] = bf16_hi(v.z());
  f[6] = bf16_lo(v.w()); f[7] = bf16_hi(v.w());
}

// A work-group has BLOCK work-items and covers BLOCK / (head_dim / 8)
// consecutive heads, which may span tokens. Each token contributes its Q heads
// followed by its K heads. Work-item g of a head owns elements [8g, 8g + 8).
//
// The head_dim / 8 work-items of a head are contiguous in one sub-group, so
// the sum of squares is reduced with xor permutes and the rotate-half partner
// (element i + head_dim/2, on work-item g + (head_dim/2)/8) is fetched with
// select_from_group. There is no work-group barrier. Work-items past the last
// head still take part in the sub-group operations and skip the stores.
template <int HEAD_DIM, int BLOCK>
static inline void qk_rmsnorm_mrope(const bf16* __restrict q,
                                    const bf16* __restrict k,
                                    bf16* __restrict q_out,
                                    bf16* __restrict k_out,
                                    const bf16* __restrict q_weight,
                                    const bf16* __restrict k_weight,
                                    const bf16* __restrict cos_sin_cache,
                                    const int64_t* __restrict positions,
                                    const int* __restrict axis_map,
                                    int tokens, int heads, int pos_stride, float eps,
                                    const sycl::nd_item<1>& item) {
  constexpr int VEC = 8;
  constexpr int G = HEAD_DIM / VEC;
  constexpr int HEADS_PER_BLOCK = BLOCK / G;
  constexpr int HALF = HEAD_DIM / 2;
  static_assert(BLOCK % G == 0, "a work-group must hold whole heads");

  const sycl::sub_group sg = item.get_sub_group();
  const int lid = item.get_local_id(0);
  const int g = lid % G;
  const int group_base = sg.get_local_linear_id() & ~(G - 1);
  const int nheads = heads * 2;
  const int total = tokens * nheads;
  const int64_t gh = static_cast<int64_t>(item.get_group(0)) * HEADS_PER_BLOCK + lid / G;
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

  const sycl::uint4 xv = *reinterpret_cast<const sycl::uint4*>(in_base + d0);
  const sycl::uint4 wv = *reinterpret_cast<const sycl::uint4*>(weight + d0);

  const int h8 = HALF / VEC;
  const bool first = g < h8;
  const int c0 = first ? d0 : d0 - HALF;

  constexpr int ROWS = 3;
  sycl::uint4 cv[ROWS], sv[ROWS];
#pragma unroll
  for (int a = 0; a < ROWS; ++a) {
    const int64_t pos = positions[static_cast<int64_t>(a) * pos_stride + token];
    const bf16* row = cos_sin_cache + pos * HEAD_DIM;
    cv[a] = *reinterpret_cast<const sycl::uint4*>(row + c0);
    sv[a] = *reinterpret_cast<const sycl::uint4*>(row + HALF + c0);
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
    ss += sycl::permute_group_by_xor(sg, ss, m);
  const float inv_rms = sycl::rsqrt(ss / static_cast<float>(HEAD_DIM) + eps);

  uint32_t n[VEC / 2];
#pragma unroll
  for (int i = 0; i < VEC / 2; ++i)
    n[i] = pack_bf16x2(x[2 * i] * inv_rms * w[2 * i],
                       x[2 * i + 1] * inv_rms * w[2 * i + 1]);

  // sycl::select_from_group always indexes the whole sub-group.
  // There is no width argument, so the slice base has to be written into the id.
  const int src = group_base + (first ? g + h8 : g - h8);
  uint32_t p[VEC / 2];
#pragma unroll
  for (int i = 0; i < VEC / 2; ++i)
    p[i] = sycl::select_from_group(sg, n[i], src);

  float c[VEC], s[VEC];
  const sycl::int4 a_lo = *reinterpret_cast<const sycl::int4*>(axis_map + c0);
  const sycl::int4 a_hi = *reinterpret_cast<const sycl::int4*>(axis_map + c0 + 4);
  const int axes[VEC] = {a_lo.x(), a_lo.y(), a_lo.z(), a_lo.w(),
                         a_hi.x(), a_hi.y(), a_hi.z(), a_hi.w()};
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
    *reinterpret_cast<sycl::uint4*>(out_base + d0) = sycl::uint4(r[0], r[1], r[2], r[3]);
}

template <int HEAD_DIM, int BLOCK, int SG>
class qk_rmsnorm_mrope_k;

template <int HEAD_DIM, int BLOCK, int SG>
static void launch_block(sycl::queue& q, const bf16* qi, const bf16* k, bf16* q_out,
                         bf16* k_out, const bf16* q_weight, const bf16* k_weight,
                         const bf16* cache, const int64_t* positions, const int* axis_map,
                         int tokens, int heads, int pos_stride, float eps) {
  constexpr int heads_per_block = BLOCK / (HEAD_DIM / 8);
  const int64_t total = static_cast<int64_t>(tokens) * heads * 2;
  const size_t blocks = static_cast<size_t>((total + heads_per_block - 1) / heads_per_block);
  q.parallel_for<qk_rmsnorm_mrope_k<HEAD_DIM, BLOCK, SG>>(
      sycl::nd_range<1>(blocks * BLOCK, BLOCK),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
        qk_rmsnorm_mrope<HEAD_DIM, BLOCK>(qi, k, q_out, k_out, q_weight, k_weight, cache,
                                          positions, axis_map, tokens, heads, pos_stride,
                                          eps, item);
      });
}

static const int block_sizes[] = {64, 128, 256, 512, 1024};

template <int HEAD_DIM, int SG>
static void launch(sycl::queue& q, const bf16* qi, const bf16* k, bf16* q_out, bf16* k_out,
                   const bf16* q_weight, const bf16* k_weight, const bf16* cache,
                   const int64_t* positions, const int* axis_map, int tokens, int heads,
                   int pos_stride, float eps, int block_size) {
  switch (block_size) {
    case 64:
      launch_block<HEAD_DIM, 64, SG>(q, qi, k, q_out, k_out, q_weight, k_weight, cache,
                                     positions, axis_map, tokens, heads, pos_stride, eps);
      break;
    case 128:
      launch_block<HEAD_DIM, 128, SG>(q, qi, k, q_out, k_out, q_weight, k_weight, cache,
                                      positions, axis_map, tokens, heads, pos_stride, eps);
      break;
    case 256:
      launch_block<HEAD_DIM, 256, SG>(q, qi, k, q_out, k_out, q_weight, k_weight, cache,
                                      positions, axis_map, tokens, heads, pos_stride, eps);
      break;
    case 512:
      launch_block<HEAD_DIM, 512, SG>(q, qi, k, q_out, k_out, q_weight, k_weight, cache,
                                      positions, axis_map, tokens, heads, pos_stride, eps);
      break;
    case 1024:
      launch_block<HEAD_DIM, 1024, SG>(q, qi, k, q_out, k_out, q_weight, k_weight, cache,
                                       positions, axis_map, tokens, heads, pos_stride, eps);
      break;
    default:
      printf("unsupported block size %d\n", block_size);
      exit(EXIT_FAILURE);
  }
}

// The largest device sub-group is the warp width passed to the kernel.
// A head uses at most 32 work-items, so a sub-group of 32 or 64 holds one.
template <typename F>
static void with_sub_group(int sub_group_size, F&& f) {
  switch (sub_group_size) {
    case 32:
      f(std::integral_constant<int, 32>{});
      break;
    case 64:
      f(std::integral_constant<int, 64>{});
      break;
    default:
      printf("unsupported sub-group size %d\n", sub_group_size);
      exit(EXIT_FAILURE);
  }
}

// Device rsqrt and the host libm sqrt do not match bit for bit, and the two
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (sg_sizes.empty()) {
    printf("device reports no sub-group sizes\n");
    return 1;
  }
  const int sub_group_size =
      static_cast<int>(*std::max_element(sg_sizes.begin(), sg_sizes.end()));
  const int lanes_per_head = head_dim / 8;
  if (sub_group_size < lanes_per_head || sub_group_size % lanes_per_head != 0) {
    printf("sub-group size %d does not hold a head of %d elements\n", sub_group_size,
           head_dim);
    return 1;
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

  bf16* d_q = sycl::malloc_device<bf16>(qk_n, q);
  bf16* d_k = sycl::malloc_device<bf16>(qk_n, q);
  bf16* d_qw = sycl::malloc_device<bf16>(head_dim, q);
  bf16* d_kw = sycl::malloc_device<bf16>(head_dim, q);
  bf16* d_cache = sycl::malloc_device<bf16>(cache_n, q);
  bf16* d_qo = sycl::malloc_device<bf16>(qk_n, q);
  bf16* d_ko = sycl::malloc_device<bf16>(qk_n, q);
  int64_t* d_pos = sycl::malloc_device<int64_t>(pos_n, q);
  int* d_axis = sycl::malloc_device<int>(half, q);
  q.memcpy(d_q, h_q.data(), qk_n * sizeof(bf16));
  q.memcpy(d_k, h_k.data(), qk_n * sizeof(bf16));
  q.memcpy(d_qw, h_qw.data(), head_dim * sizeof(bf16));
  q.memcpy(d_kw, h_kw.data(), head_dim * sizeof(bf16));
  q.memcpy(d_cache, h_cache.data(), cache_n * sizeof(bf16));
  q.memcpy(d_pos, h_pos.data(), pos_n * sizeof(int64_t));
  q.memcpy(d_axis, axis_map.data(), half * sizeof(int));
  q.wait();

  const int pos_stride = tokens;
  auto launch_kernel = [&](int block_size) {
    with_sub_group(sub_group_size, [&](auto sg) {
      constexpr int SG = decltype(sg)::value;
      if (head_dim == 64)
        launch<64, SG>(q, d_q, d_k, d_qo, d_ko, d_qw, d_kw, d_cache, d_pos, d_axis, tokens,
                       heads, pos_stride, eps, block_size);
      else if (head_dim == 128)
        launch<128, SG>(q, d_q, d_k, d_qo, d_ko, d_qw, d_kw, d_cache, d_pos, d_axis, tokens,
                        heads, pos_stride, eps, block_size);
      else
        launch<256, SG>(q, d_q, d_k, d_qo, d_ko, d_qw, d_kw, d_cache, d_pos, d_axis, tokens,
                        heads, pos_stride, eps, block_size);
    });
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
    q.memset(d_qo, 0, qk_n * sizeof(bf16));
    q.memset(d_ko, 0, qk_n * sizeof(bf16));
    launch_kernel(block_size);
    q.memcpy(h_qo.data(), d_qo, qk_n * sizeof(bf16));
    q.memcpy(h_ko.data(), d_ko, qk_n * sizeof(bf16));
    q.wait();
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
      q.wait();

      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; ++i)
        launch_kernel(block_size);
      q.wait();
      auto stop = std::chrono::steady_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(stop - start).count() / repeat;
      const double gbps = (bytes / ms) / 1e6;
      printf("block_size %4d | Average kernel execution time: %f (ms) | bandwidth %.2f GB/s\n",
             block_size, ms, gbps);
    }
  }

  sycl::free(d_q, q);
  sycl::free(d_k, q);
  sycl::free(d_qw, q);
  sycl::free(d_kw, q);
  sycl::free(d_cache, q);
  sycl::free(d_qo, q);
  sycl::free(d_ko, q);
  sycl::free(d_pos, q);
  sycl::free(d_axis, q);

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
