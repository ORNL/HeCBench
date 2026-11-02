#pragma once

// SYCL kernels for the Mamba-2 SSD forward pass. The host selects joint_matrix,
// the register-blocked GEMM path, or the scalar tiled/global paths.

#include <climits>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include "reference.h"

namespace matrix = sycl::ext::oneapi::experimental::matrix;

// BF16 16x16x16 joint_matrix maps to mma.sync on NVIDIA sm_80+ and to MFMA
// where this DPC++ provides the HIP backend. AMD wavefront size is 32 or 64;
// launches use the largest device-reported subgroup size.
// The TF32 16x8 shape is NVIDIA-only, so AMD images apply that product with
// TF32-rounded scalar FMAs. matrix-hip.hpp defines __HIP_PLATFORM_AMD_MFMA__
// for the targets it implements; this DPC++ does that for the gfx90a
// 16x16x16bf16 MFMA and not for gfx908, whose BF16 MFMA shapes are different.
#if defined(__SYCL_DEVICE_ONLY__) &&                                           \
    !(defined(__HIP_PLATFORM_AMD_MFMA__) ||                                    \
      (defined(__NVPTX__) && __SYCL_CUDA_ARCH__ >= 800))
#define SSM2_MMA_DEVICE 0
#else
#define SSM2_MMA_DEVICE 1
#endif

using sycl_bfloat16 = sycl::ext::oneapi::bfloat16;

// joint_matrix lowers to gfx90a MFMA 16x16x16bf16_1k. That instruction wants
// four consecutive K values in each lane, which this HIP backend loads with
// layout::col_major when A lives at m * ld + k and layout::row_major when A
// lives at k * ld + m.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
constexpr matrix::layout kMmaARow = matrix::layout::col_major;
constexpr matrix::layout kMmaACol = matrix::layout::row_major;
#else
constexpr matrix::layout kMmaARow = matrix::layout::row_major;
constexpr matrix::layout kMmaACol = matrix::layout::col_major;
#endif

template <typename T> static inline auto matrix_global_ptr(T *p) {
  return sycl::address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(p);
}

template <typename T> static inline auto matrix_local_ptr(T *p) {
  return sycl::address_space_cast<sycl::access::address_space::local_space,
                                  sycl::access::decorated::no>(p);
}

constexpr float kLog2e = 1.4426950408889634f;
constexpr int kThreads = 256;
constexpr int kStatePTile = 16;
constexpr int kStateNTile = 16;
constexpr int kOutPTile = 8;

// 16-byte vector I/O matching the CUDA float4 loads. The GEMM planner only
// admits extents that are multiples of 4, so each address below is aligned.
struct alignas(16) Float4 {
  float x, y, z, w;
};

static inline Float4 ld4(const float *p) {
  return *reinterpret_cast<const Float4 *>(p);
}

static inline void st4(float *p, Float4 value) {
  *reinterpret_cast<Float4 *>(p) = value;
}

static inline void st4(float *p, float a, float b, float c, float d) {
  st4(p, Float4{a, b, c, d});
}

// Widen four BF16 values from global memory into one Float4 local slot.
static inline void stage4(float *dst, const ssm2_bf16 *src) {
  float v[4];
  ssm2_ld4_bf16(src, v);
  st4(dst, v[0], v[1], v[2], v[3]);
}

// Storage types follow ssm2_types.h: BF16 activations and chunk-entry states,
// FP32 parameters, decay, carried state, and the C @ B^T workspace.
struct Params {
  int B, T, H, G, P, N, chunk, nchunks, heads_per_group;
  const ssm2_bf16 *x;
  const float *a;
  const ssm2_bf16 *b, *c;
  const float *dskip;
  const ssm2_bf16 *dt;
  const float *init_state;
  float *decay, *cb;
  ssm2_bf16 *hiddens;
  float *final_state;
  ssm2_bf16 *out;
};

static inline size_t cb_idx(const Params &p, int b, int chunk, int group, int i,
                            int j) {
  return (((size_t)b * (size_t)p.nchunks + (size_t)chunk) * (size_t)p.G +
          (size_t)group) *
             (size_t)p.chunk * (size_t)p.chunk +
         (size_t)i * (size_t)p.chunk + (size_t)j;
}

// Cap a logical work-group count at the device's one-dimensional work-group
// count limit (the SYCL counterpart of cudaDevAttrMaxGridDimX). Unlike CUDA
// and HIP, SYCL expresses a launch as a total global range, so also cap the
// product of groups and local size to the range the runtime accepts.
static size_t grid_groups(size_t blocks, size_t max_groups, size_t local) {
  const size_t range_groups = (size_t)INT_MAX / local;
  const size_t limit = max_groups < range_groups ? max_groups : range_groups;
  return blocks < limit ? blocks : limit;
}

static void launch_cumsum(sycl::queue &q, const Params &p, size_t max_groups) {
  const size_t blocks = (size_t)p.B * p.nchunks * p.H;
  const size_t local = (size_t)p.chunk;
  const size_t groups = grid_groups(blocks, max_groups, local);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> values(p.chunk, h);
    h.parallel_for(
        sycl::nd_range<1>(groups * local, local), [=](sycl::nd_item<1> item) {
          const int tid = item.get_local_id(0);
          const auto work_group = item.get_group();
          const size_t stride = work_group.get_group_range(0);
          for (size_t linear_block = work_group.get_group_id()[0]; linear_block < blocks;
               linear_block += stride) {
            size_t block = linear_block;
            const int head = block % (size_t)p.H;
            block /= (size_t)p.H;
            const int chunk = block % (size_t)p.nchunks;
            const int batch = block / (size_t)p.nchunks;
            const int t = chunk * p.chunk + tid;
            values[tid] = p.a[head] * ssm2_from_bf16(
                                          p.dt[ssm2_dt_idx(batch, t, head, p.T,
                                                           p.H)]);
            sycl::group_barrier(work_group);
            for (int offset = 1; offset < p.chunk; offset <<= 1) {
              const float add = tid >= offset ? values[tid - offset] : 0.f;
              sycl::group_barrier(work_group);
              values[tid] += add;
              sycl::group_barrier(work_group);
            }
            p.decay[ssm2_dt_idx(batch, t, head, p.T, p.H)] =
                values[tid] * kLog2e;
            sycl::group_barrier(work_group);
          }
        });
  });
}

// Advance batch/head states directly from global memory using a group stride.
static void launch_state_global(sycl::queue &q, const Params &p,
                                size_t max_groups, int threads) {
  const size_t blocks = (size_t)p.B * p.H;
  const size_t local = (size_t)threads;
  const size_t groups = grid_groups(blocks, max_groups, local);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(groups * local, local), [=](sycl::nd_item<1> item) {
          const int tid = item.get_local_id(0);
          const auto work_group = item.get_group();
          const size_t stride = work_group.get_group_range(0);
          const size_t step = item.get_local_range(0);
          for (size_t bh = work_group.get_group_id()[0]; bh < blocks; bh += stride) {
            const int batch = bh / (size_t)p.H;
            const int head = bh - (size_t)batch * p.H;
            const int group = head / p.heads_per_group;
            const size_t PN = (size_t)p.P * (size_t)p.N;
            for (size_t i = tid; i < PN; i += step) {
              const int hp = (int)(i / (size_t)p.N);
              const int n = (int)(i - (size_t)hp * (size_t)p.N);
              p.final_state[ssm2_st_idx(batch, head, hp, n, p.H, p.P, p.N)] =
                  p.init_state[ssm2_st_idx(batch, head, hp, n, p.H, p.P, p.N)];
            }
            sycl::group_barrier(work_group);
            for (int chunk = 0; chunk < p.nchunks; ++chunk) {
              const int t0 = chunk * p.chunk;
              for (size_t i = tid; i < PN; i += step) {
                const int hp = (int)(i / (size_t)p.N);
                const int n = (int)(i - (size_t)hp * (size_t)p.N);
                p.hiddens[ssm2_hid_idx(batch, head, chunk, hp, n, p.H,
                                       p.nchunks, p.P, p.N)] =
                    ssm2_to_bf16(p.final_state[ssm2_st_idx(batch, head, hp, n,
                                                           p.H, p.P, p.N)]);
              }
              sycl::group_barrier(work_group);
              const float dlast =
                  p.decay[ssm2_dt_idx(batch, t0 + p.chunk - 1, head, p.T, p.H)];
              const float scale = sycl::exp2(dlast);
              for (size_t i = tid; i < PN; i += step) {
                const int hp = (int)(i / (size_t)p.N);
                const int n = (int)(i - (size_t)hp * (size_t)p.N);
                float acc =
                    scale * p.final_state[ssm2_st_idx(batch, head, hp, n, p.H,
                                                      p.P, p.N)];
                for (int t = 0; t < p.chunk; ++t) {
                  const int gt = t0 + t;
                  const float dec =
                      p.decay[ssm2_dt_idx(batch, gt, head, p.T, p.H)];
                  const float dtv = ssm2_from_bf16(
                      p.dt[ssm2_dt_idx(batch, gt, head, p.T, p.H)]);
                  const float xv = ssm2_from_bf16(
                      p.x[ssm2_x_idx(batch, gt, head, hp, p.T, p.H, p.P)]);
                  const float bv = ssm2_from_bf16(
                      p.b[ssm2_bc_idx(batch, gt, group, n, p.T, p.G, p.N)]);
                  acc += xv * ssm2_round_bf16(
                                  bv * (dtv * sycl::exp2(dlast - dec)));
                }
                p.final_state[ssm2_st_idx(batch, head, hp, n, p.H, p.P, p.N)] =
                    acc;
              }
              sycl::group_barrier(work_group);
            }
          }
        });
  });
}

// Advance tiled P x N state planes so small B*H shapes expose more parallelism.
// One work-item owns one element of the 16x16 tile, so the group size is 256.
static void launch_state_tiled(sycl::queue &q, const Params &p,
                               size_t max_groups, size_t smem_bytes) {
  const int ptiles = (p.P + kStatePTile - 1) / kStatePTile;
  const int ntiles = (p.N + kStateNTile - 1) / kStateNTile;
  const size_t blocks = (size_t)p.B * p.H * ptiles * ntiles;
  const size_t local = (size_t)kThreads;
  const size_t groups = grid_groups(blocks, max_groups, local);
  const size_t smem_elems = smem_bytes / sizeof(float);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> sm(smem_elems, h);
    h.parallel_for(
        sycl::nd_range<1>(groups * local, local), [=](sycl::nd_item<1> item) {
          const int tid = item.get_local_id(0);
          const auto work_group = item.get_group();
          const size_t stride = work_group.get_group_range(0);
          const size_t x_off = 0;
          const size_t b_off = (size_t)p.chunk * kStatePTile;
          const int lp = tid / kStateNTile;
          const int ln = tid % kStateNTile;
          for (size_t linear_block = work_group.get_group_id()[0]; linear_block < blocks;
               linear_block += stride) {
            size_t block = linear_block;
            const int ntile = block % (size_t)ntiles;
            block /= (size_t)ntiles;
            const int ptile = block % (size_t)ptiles;
            block /= (size_t)ptiles;
            const int head = block % (size_t)p.H;
            const int batch = block / (size_t)p.H;
            const int group = head / p.heads_per_group;
            const int p0 = ptile * kStatePTile;
            const int n0 = ntile * kStateNTile;
            const int hp = p0 + lp;
            const int sn = n0 + ln;
            const bool valid = hp < p.P && sn < p.N;
            if (valid)
              p.final_state[ssm2_st_idx(batch, head, hp, sn, p.H, p.P, p.N)] =
                  p.init_state[ssm2_st_idx(batch, head, hp, sn, p.H, p.P, p.N)];
            for (int chunk = 0; chunk < p.nchunks; ++chunk) {
              const int t0 = chunk * p.chunk;
              if (valid)
                p.hiddens[ssm2_hid_idx(batch, head, chunk, hp, sn, p.H,
                                       p.nchunks, p.P, p.N)] =
                    ssm2_to_bf16(p.final_state[ssm2_st_idx(batch, head, hp, sn,
                                                           p.H, p.P, p.N)]);
              for (int i = tid; i < p.chunk * kStatePTile; i += kThreads) {
                const int t = i / kStatePTile;
                const int pp = p0 + i % kStatePTile;
                sm[x_off + i] =
                    pp < p.P ? ssm2_from_bf16(p.x[ssm2_x_idx(
                                   batch, t0 + t, head, pp, p.T, p.H, p.P)])
                             : 0.f;
              }
              const float dlast =
                  p.decay[ssm2_dt_idx(batch, t0 + p.chunk - 1, head, p.T, p.H)];
              // Stage B already weighted by dt * exp2(d_last - d) and rounded
              // to BF16.
              for (int i = tid; i < p.chunk * kStateNTile; i += kThreads) {
                const int t = i / kStateNTile;
                const int nn = n0 + i % kStateNTile;
                float bw = 0.f;
                if (nn < p.N) {
                  const float dec =
                      p.decay[ssm2_dt_idx(batch, t0 + t, head, p.T, p.H)];
                  const float dtv = ssm2_from_bf16(
                      p.dt[ssm2_dt_idx(batch, t0 + t, head, p.T, p.H)]);
                  const float bv = ssm2_from_bf16(p.b[ssm2_bc_idx(
                      batch, t0 + t, group, nn, p.T, p.G, p.N)]);
                  bw = ssm2_round_bf16(
                      bv * (dtv * sycl::exp2(dlast - dec)));
                }
                sm[b_off + i] = bw;
              }
              sycl::group_barrier(work_group);
              if (valid) {
                float acc = sycl::exp2(dlast) *
                            p.final_state[ssm2_st_idx(batch, head, hp, sn, p.H,
                                                      p.P, p.N)];
#pragma unroll 4
                for (int t = 0; t < p.chunk; ++t)
                  acc += sm[x_off + t * kStatePTile + lp] *
                         sm[b_off + t * kStateNTile + ln];
                p.final_state[ssm2_st_idx(batch, head, hp, sn, p.H, p.P, p.N)] =
                    acc;
              }
              sycl::group_barrier(work_group);
            }
          }
        });
  });
}

// Materialize grouped C @ B^T matrices for all batches and chunks.
static void launch_cb(sycl::queue &q, const Params &p, size_t max_groups,
                      int threads) {
  const size_t blocks = (size_t)p.B * p.nchunks * p.G;
  const size_t local = (size_t)threads;
  const size_t groups = grid_groups(blocks, max_groups, local);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(groups * local, local), [=](sycl::nd_item<1> item) {
          const int tid = item.get_local_id(0);
          const auto work_group = item.get_group();
          const size_t stride = work_group.get_group_range(0);
          for (size_t linear_block = work_group.get_group_id()[0]; linear_block < blocks;
               linear_block += stride) {
            size_t block = linear_block;
            const int group = block % (size_t)p.G;
            block /= (size_t)p.G;
            const int chunk = block % (size_t)p.nchunks;
            const int batch = block / (size_t)p.nchunks;
            const int t0 = chunk * p.chunk;
            for (int ij = tid; ij < p.chunk * p.chunk; ij += threads) {
              const int i = ij / p.chunk;
              const int j = ij - i * p.chunk;
              float dot = 0.f;
              for (int n = 0; n < p.N; ++n)
                dot += ssm2_from_bf16(p.c[ssm2_bc_idx(batch, t0 + i, group, n,
                                                      p.T, p.G, p.N)]) *
                       ssm2_from_bf16(p.b[ssm2_bc_idx(batch, t0 + j, group, n,
                                                      p.T, p.G, p.N)]);
              p.cb[cb_idx(p, batch, chunk, group, i, j)] = dot;
            }
          }
        });
  });
}

// Produce output for one head-dimension tile using shared input and C @ B^T.
static void launch_out_tiled(sycl::queue &q, const Params &p, size_t max_groups,
                             int threads, size_t smem_bytes) {
  const int ptiles = (p.P + kOutPTile - 1) / kOutPTile;
  const size_t blocks = (size_t)p.B * p.nchunks * p.H * ptiles;
  const size_t local = (size_t)threads;
  const size_t groups = grid_groups(blocks, max_groups, local);
  const size_t smem_elems = smem_bytes / sizeof(float);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> sm(smem_elems, h);
    h.parallel_for(
        sycl::nd_range<1>(groups * local, local), [=](sycl::nd_item<1> item) {
          const int tid = item.get_local_id(0);
          const auto work_group = item.get_group();
          const size_t stride = work_group.get_group_range(0);
          for (size_t linear_block = work_group.get_group_id()[0]; linear_block < blocks;
               linear_block += stride) {
            size_t block = linear_block;
            const int ptile = block % (size_t)ptiles;
            block /= (size_t)ptiles;
            const int head = block % (size_t)p.H;
            block /= (size_t)p.H;
            const int chunk = block % (size_t)p.nchunks;
            const int batch = block / (size_t)p.nchunks;
            const int group = head / p.heads_per_group;
            const int p0 = ptile * kOutPTile;
            const int t0 = chunk * p.chunk;
            const size_t dec_off = 0;
            const size_t dtx_off = p.chunk;
            const size_t cb_off = dtx_off + (size_t)p.chunk * kOutPTile;
            for (int t = tid; t < p.chunk; t += threads)
              sm[dec_off + t] =
                  p.decay[ssm2_dt_idx(batch, t0 + t, head, p.T, p.H)];
            // BF16 values are exact in TF32, so bf16(dt * x) is already the
            // operand.
            for (int i = tid; i < p.chunk * kOutPTile; i += threads) {
              const int t = i / kOutPTile;
              const int hp = p0 + i % kOutPTile;
              float dtx = 0.f;
              if (hp < p.P)
                dtx = ssm2_round_bf16(
                    ssm2_from_bf16(
                        p.dt[ssm2_dt_idx(batch, t0 + t, head, p.T, p.H)]) *
                    ssm2_from_bf16(p.x[ssm2_x_idx(batch, t0 + t, head, hp, p.T,
                                                  p.H, p.P)]));
              sm[dtx_off + i] = dtx;
            }
            for (int ij = tid; ij < p.chunk * p.chunk; ij += threads) {
              const int i = ij / p.chunk;
              const int j = ij - i * p.chunk;
              sm[cb_off + ij] = p.cb[cb_idx(p, batch, chunk, group, i, j)];
            }
            sycl::group_barrier(work_group);
            const float skip = p.dskip[head];
            for (int ip = tid; ip < p.chunk * kOutPTile; ip += threads) {
              const int t = ip / kOutPTile;
              const int lp = ip % kOutPTile;
              const int hp = p0 + lp;
              if (hp >= p.P)
                continue;
              float carried = 0.f;
              for (int n = 0; n < p.N; ++n)
                carried += ssm2_from_bf16(p.c[ssm2_bc_idx(batch, t0 + t, group,
                                                          n, p.T, p.G, p.N)]) *
                           ssm2_from_bf16(p.hiddens[ssm2_hid_idx(
                               batch, head, chunk, hp, n, p.H, p.nchunks, p.P,
                               p.N)]);
              float acc = carried * sycl::exp2(sm[dec_off + t]);
              for (int j = 0; j <= t; ++j)
                acc += ssm2_round_tf32(sm[cb_off + t * p.chunk + j] *
                                       sycl::exp2(sm[dec_off + t] -
                                                          sm[dec_off + j])) *
                       sm[dtx_off + j * kOutPTile + lp];
              const size_t idx =
                  ssm2_x_idx(batch, t0 + t, head, hp, p.T, p.H, p.P);
              acc += skip * ssm2_from_bf16(p.x[idx]);
              p.out[idx] = ssm2_to_bf16(acc);
            }
            sycl::group_barrier(work_group);
          }
        });
  });
}

// Produce output directly from global memory using a group stride.
static void launch_out_global(sycl::queue &q, const Params &p, size_t max_groups,
                              int threads) {
  const size_t blocks = (size_t)p.B * p.nchunks * p.H;
  const size_t local = (size_t)threads;
  const size_t groups = grid_groups(blocks, max_groups, local);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(groups * local, local), [=](sycl::nd_item<1> item) {
          const int tid = item.get_local_id(0);
          const auto work_group = item.get_group();
          const size_t stride = work_group.get_group_range(0);
          for (size_t linear_block = work_group.get_group_id()[0]; linear_block < blocks;
               linear_block += stride) {
            size_t block = linear_block;
            const int head = block % (size_t)p.H;
            block /= (size_t)p.H;
            const int chunk = block % (size_t)p.nchunks;
            const int batch = block / (size_t)p.nchunks;
            const int group = head / p.heads_per_group;
            const int t0 = chunk * p.chunk;
            const float skip = p.dskip[head];
            for (int ip = tid; ip < p.chunk * p.P; ip += threads) {
              const int t = ip / p.P;
              const int hp = ip - t * p.P;
              const int gt = t0 + t;
              const float dec_t =
                  p.decay[ssm2_dt_idx(batch, gt, head, p.T, p.H)];
              const float xv = ssm2_from_bf16(
                  p.x[ssm2_x_idx(batch, gt, head, hp, p.T, p.H, p.P)]);
              float carried = 0.f;
              for (int n = 0; n < p.N; ++n)
                carried += ssm2_from_bf16(p.c[ssm2_bc_idx(batch, gt, group, n,
                                                          p.T, p.G, p.N)]) *
                           ssm2_from_bf16(p.hiddens[ssm2_hid_idx(
                               batch, head, chunk, hp, n, p.H, p.nchunks, p.P,
                               p.N)]);
              float acc = carried * sycl::exp2(dec_t);
              for (int j = 0; j <= t; ++j) {
                const int gj = t0 + j;
                float dot = 0.f;
                for (int n = 0; n < p.N; ++n)
                  dot += ssm2_from_bf16(p.c[ssm2_bc_idx(batch, gt, group, n,
                                                        p.T, p.G, p.N)]) *
                         ssm2_from_bf16(p.b[ssm2_bc_idx(batch, gj, group, n,
                                                        p.T, p.G, p.N)]);
                const float dec_j =
                    p.decay[ssm2_dt_idx(batch, gj, head, p.T, p.H)];
                const float dtv = ssm2_from_bf16(
                    p.dt[ssm2_dt_idx(batch, gj, head, p.T, p.H)]);
                const float xj = ssm2_from_bf16(
                    p.x[ssm2_x_idx(batch, gj, head, hp, p.T, p.H, p.P)]);
                acc += ssm2_round_tf32(dot * sycl::exp2(dec_t - dec_j)) *
                       ssm2_round_bf16(dtv * xj);
              }
              acc += skip * xv;
              p.out[ssm2_x_idx(batch, gt, head, hp, p.T, p.H, p.P)] =
                  ssm2_to_bf16(acc);
            }
          }
        });
  });
}

// Register-blocked 4x4 C @ B^T for one chunk of one group.
static void launch_gemm_cb(sycl::queue &q, const Params &p, size_t max_groups,
                           size_t cb_smem) {
  const int cb_threads = (p.chunk / 4) * (p.chunk / 4);
  const size_t cb_blocks = (size_t)p.B * p.nchunks * p.G;
  const size_t cb_groups =
      grid_groups(cb_blocks, max_groups, (size_t)cb_threads);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> sm(cb_smem / sizeof(float), h);
    h.parallel_for(
        sycl::nd_range<1>(cb_groups * (size_t)cb_threads, (size_t)cb_threads),
        [=](sycl::nd_item<1> item) {
          const int local = item.get_local_id(0);
          const auto work_group = item.get_group();
          float *smem = sm.get_multi_ptr<sycl::access::decorated::no>().get();
          const int chunk = p.chunk;
          const int tiles = chunk / 4;
          const int row = (local / tiles) * 4;
          const int col = (local % tiles) * 4;
          const size_t stride = work_group.get_group_range(0);
          for (size_t linear_block = work_group.get_group_id()[0];
               linear_block < cb_blocks; linear_block += stride) {
            size_t block = linear_block;
            const int group = block % (size_t)p.G;
            block /= (size_t)p.G;
            const int chunk_idx = block % (size_t)p.nchunks;
            const int batch = block / (size_t)p.nchunks;
            const int t0 = chunk_idx * chunk;
            float acc[4][4] = {};
            float *cs = smem;
            float *bs = cs + 4 * chunk;
            for (int n0 = 0; n0 < p.N; n0 += 4) {
              for (int t = local; t < chunk; t += cb_threads) {
                stage4(cs + t * 4, p.c + ssm2_bc_idx(batch, t0 + t, group, n0,
                                                     p.T, p.G, p.N));
                stage4(bs + t * 4, p.b + ssm2_bc_idx(batch, t0 + t, group, n0,
                                                     p.T, p.G, p.N));
              }
              sycl::group_barrier(work_group);
              Float4 creg[4], breg[4];
#pragma unroll
              for (int i = 0; i < 4; ++i)
                creg[i] = ld4(cs + (row + i) * 4);
#pragma unroll
              for (int j = 0; j < 4; ++j)
                breg[j] = ld4(bs + (col + j) * 4);
#pragma unroll
              for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                  acc[i][j] += creg[i].x * breg[j].x;
                  acc[i][j] += creg[i].y * breg[j].y;
                  acc[i][j] += creg[i].z * breg[j].z;
                  acc[i][j] += creg[i].w * breg[j].w;
                }
              sycl::group_barrier(work_group);
            }
            float *dst = p.cb + cb_idx(p, batch, chunk_idx, group, row, col);
#pragma unroll
            for (int i = 0; i < 4; ++i)
              st4(dst + i * chunk, acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            sycl::group_barrier(work_group);
          }
        });
  });
}

// oneAPI SPIR-V exposes BF16 joint_matrix as 8x16x16 on a 16-lane subgroup,
// not the 16x16x16 shape of the CUDA and AMD kernels in launch_mma. The GEMM
// path uses it for C @ B^T; its state and output kernels consume the result.
// Kernel captures are laid out in first-use order, which differs between the
// device body and the host's #else branch. One captured struct gives both
// passes the same layout.
struct CbJointMatrixArgs {
  Params p;
  int row_tiles, col_tiles;
  size_t tile_blocks;
};

template <int Sg>
static void launch_cb_joint_matrix_8x16(sycl::queue &q, const Params &p,
                                        size_t max_groups) {
  const size_t blocks = (size_t)p.B * p.nchunks * p.G;
  const CbJointMatrixArgs args{p, p.chunk / 8, p.chunk / 16,
                               blocks * (size_t)(p.chunk / 8) *
                                   (size_t)(p.chunk / 16)};
  const size_t groups = grid_groups(args.tile_blocks, max_groups, Sg);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(groups * (size_t)Sg, (size_t)Sg),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(Sg)]] {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
          const Params &p = args.p;
          const int row_tiles = args.row_tiles;
          const int col_tiles = args.col_tiles;
          const size_t tile_blocks = args.tile_blocks;
          const sycl::sub_group sg = item.get_sub_group();
          const auto work_group = item.get_group();
          const size_t stride = work_group.get_group_range(0);
          const size_t tile_pairs = (size_t)row_tiles * (size_t)col_tiles;
          for (size_t linear = work_group.get_group_id()[0];
               linear < tile_blocks; linear += stride) {
            const int tile = (int)(linear % tile_pairs);
            size_t block = linear / tile_pairs;
            const int group = block % (size_t)p.G;
            block /= (size_t)p.G;
            const int chunk_idx = block % (size_t)p.nchunks;
            const int batch = block / (size_t)p.nchunks;
            const int t0 = chunk_idx * p.chunk;
            const int row = (tile / col_tiles) * 8;
            const int col = (tile % col_tiles) * 16;
            matrix::joint_matrix<sycl::sub_group, float,
                                 matrix::use::accumulator, 8, 16>
                acc;
            matrix::joint_matrix_fill(sg, acc, 0.f);
            for (int n0 = 0; n0 < p.N; n0 += 16) {
              matrix::joint_matrix<sycl::sub_group, sycl_bfloat16,
                                   matrix::use::a, 8, 16,
                                   matrix::layout::row_major>
                  c_frag;
              matrix::joint_matrix<sycl::sub_group, sycl_bfloat16,
                                   matrix::use::b, 16, 16,
                                   matrix::layout::col_major>
                  b_frag;
              const auto *c_ptr = reinterpret_cast<const sycl_bfloat16 *>(
                  p.c + ssm2_bc_idx(batch, t0 + row, group, n0, p.T, p.G,
                                    p.N));
              const auto *b_ptr = reinterpret_cast<const sycl_bfloat16 *>(
                  p.b + ssm2_bc_idx(batch, t0 + col, group, n0, p.T, p.G,
                                    p.N));
              matrix::joint_matrix_load(sg, c_frag, matrix_global_ptr(c_ptr),
                                        p.G * p.N);
              matrix::joint_matrix_load(sg, b_frag, matrix_global_ptr(b_ptr),
                                        p.G * p.N);
              matrix::joint_matrix_mad(sg, acc, c_frag, b_frag, acc);
            }
            matrix::joint_matrix_store(
                sg, acc,
                matrix_global_ptr(
                    p.cb + cb_idx(p, batch, chunk_idx, group, row, col)),
                p.chunk, matrix::layout::row_major);
          }
#else
          (void)item, (void)args;
#endif
        });
  });
}

// With use_mma, Intel GPUs take C @ B^T from the 8x16x16 joint_matrix.
// NVIDIA and AMD leave use_mma false here and use launch_mma instead.
static void launch_gemm(sycl::queue &q, const Params &p, bool use_mma,
                        int state_bh, int state_bs, int state_splits,
                        int out_bh, size_t max_groups, size_t fwd_smem,
                        size_t cb_smem, size_t out_smem) {
  const int state_tile_threads = (state_bh / 4) * (state_bs / 4);
  const int state_threads = state_tile_threads * state_splits;
  const size_t state_blocks =
      (size_t)p.B * p.H * (p.P / state_bh) * (p.N / state_bs);
  const size_t state_local = fwd_smem / sizeof(float);
  const size_t state_groups =
      grid_groups(state_blocks, max_groups, (size_t)state_threads);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> sm(state_local, h);
    h.parallel_for(
        sycl::nd_range<1>(state_groups * (size_t)state_threads,
                          (size_t)state_threads),
        [=](sycl::nd_item<1> item) {
          const int local = item.get_local_id(0);
          const auto work_group = item.get_group();
          float *smem = sm.get_multi_ptr<sycl::access::decorated::no>().get();
          const int split = local / state_tile_threads;
          const int tid = local - split * state_tile_threads;
          const int tiles_c = state_bs / 4;
          const int row = (tid / tiles_c) * 4;
          const int col = (tid % tiles_c) * 4;
          const size_t stride = work_group.get_group_range(0);
          for (size_t linear_block = work_group.get_group_id()[0];
               linear_block < state_blocks; linear_block += stride) {
            size_t block = linear_block;
            const int ntiles = p.N / state_bs;
            const int ptiles = p.P / state_bh;
            const int ntile = block % (size_t)ntiles;
            block /= (size_t)ntiles;
            const int ptile = block % (size_t)ptiles;
            block /= (size_t)ptiles;
            const int head = block % (size_t)p.H;
            const int batch = block / (size_t)p.H;
            const int group = head / p.heads_per_group;
            const int pbase = ptile * state_bh;
            const int nbase = ntile * state_bs;
            const int prow = pbase + row;
            const int ncol = nbase + col;
            float *xs = smem;
            float *bs = xs + p.chunk * state_bh;
            float *parts = bs + p.chunk * state_bs;
            float state[4][4] = {};
            if (split == 0) {
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                const Float4 v =
                    ld4(p.init_state + ssm2_st_idx(batch, head, prow + i, ncol,
                                                   p.H, p.P, p.N));
                state[i][0] = v.x;
                state[i][1] = v.y;
                state[i][2] = v.z;
                state[i][3] = v.w;
              }
            }

            for (int chunk = 0; chunk < p.nchunks; ++chunk) {
              const int t0 = chunk * p.chunk;
              if (split == 0) {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                  ssm2_st4_bf16(p.hiddens + ssm2_hid_idx(batch, head, chunk,
                                                         prow + i, ncol, p.H,
                                                         p.nchunks, p.P, p.N),
                                state[i][0], state[i][1], state[i][2],
                                state[i][3]);
              }

              for (int i = local; i < p.chunk * state_bh / 4;
                   i += state_threads) {
                const int elem = i * 4;
                const int t = elem / state_bh;
                const int lp = elem - t * state_bh;
                stage4(xs + elem, p.x + ssm2_x_idx(batch, t0 + t, head,
                                                   pbase + lp, p.T, p.H, p.P));
              }
              const float dlast = p.decay[ssm2_dt_idx(
                  batch, t0 + p.chunk - 1, head, p.T, p.H)];
              // Stage B already weighted by dt * exp2(d_last - d) and rounded
              // to BF16.
              for (int i = local; i < p.chunk * state_bs / 4;
                   i += state_threads) {
                const int elem = i * 4;
                const int t = elem / state_bs;
                const int ln = elem - t * state_bs;
                const float dec =
                    p.decay[ssm2_dt_idx(batch, t0 + t, head, p.T, p.H)];
                const float dtv = ssm2_from_bf16(
                    p.dt[ssm2_dt_idx(batch, t0 + t, head, p.T, p.H)]);
                const float w = dtv * sycl::exp2(dlast - dec);
                float bv[4];
                ssm2_ld4_bf16(p.b + ssm2_bc_idx(batch, t0 + t, group,
                                                nbase + ln, p.T, p.G, p.N),
                              bv);
                st4(bs + elem, ssm2_round_bf16(bv[0] * w),
                    ssm2_round_bf16(bv[1] * w), ssm2_round_bf16(bv[2] * w),
                    ssm2_round_bf16(bv[3] * w));
              }
              sycl::group_barrier(work_group);

              float part[4][4] = {};
              const int t_lo = split * (p.chunk / state_splits);
              const int t_hi = t_lo + p.chunk / state_splits;
              for (int t = t_lo; t < t_hi; ++t) {
                float xv[4], bv[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                  xv[i] = xs[t * state_bh + row + i];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                  bv[j] = bs[t * state_bs + col + j];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                  for (int j = 0; j < 4; ++j)
                    part[i][j] += xv[i] * bv[j];
              }
              float *mine = parts + (split * state_tile_threads + tid) * 16;
#pragma unroll
              for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                  mine[i * 4 + j] = part[i][j];
              sycl::group_barrier(work_group);

              if (split == 0) {
                const float scale = sycl::exp2(dlast);
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                  for (int j = 0; j < 4; ++j)
                    state[i][j] *= scale;
                for (int s = 0; s < state_splits; ++s) {
                  const float *src =
                      parts + (s * state_tile_threads + tid) * 16;
#pragma unroll
                  for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                      state[i][j] += src[i * 4 + j];
                }
              }
              sycl::group_barrier(work_group);
            }
            if (split == 0) {
#pragma unroll
              for (int i = 0; i < 4; ++i)
                st4(p.final_state + ssm2_st_idx(batch, head, prow + i, ncol,
                                                p.H, p.P, p.N),
                    state[i][0], state[i][1], state[i][2], state[i][3]);
            }
            sycl::group_barrier(work_group);
          }
        });
  });

  if (use_mma)
    launch_cb_joint_matrix_8x16<16>(q, p, max_groups);
  else
    launch_gemm_cb(q, p, max_groups, cb_smem);

  const int out_threads = (p.chunk / 4) * (out_bh / 4);
  const size_t out_blocks = (size_t)p.B * p.H * p.nchunks * (p.P / out_bh);
  const size_t out_groups =
      grid_groups(out_blocks, max_groups, (size_t)out_threads);
  const size_t out_local = out_smem / sizeof(float);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> sm(out_local, h);
    h.parallel_for(
        sycl::nd_range<1>(out_groups * (size_t)out_threads,
                          (size_t)out_threads),
        [=](sycl::nd_item<1> item) {
          const int local = item.get_local_id(0);
          const auto work_group = item.get_group();
          float *smem = sm.get_multi_ptr<sycl::access::decorated::no>().get();
          const int chunk = p.chunk;
          const int tiles_c = out_bh / 4;
          const int row = (local / tiles_c) * 4;
          const int col = (local % tiles_c) * 4;
          const size_t stride = work_group.get_group_range(0);
          for (size_t linear_block = work_group.get_group_id()[0];
               linear_block < out_blocks; linear_block += stride) {
            size_t block = linear_block;
            const int ptile = block % (size_t)(p.P / out_bh);
            block /= (size_t)(p.P / out_bh);
            const int chunk_idx = block % (size_t)p.nchunks;
            const size_t bh = block / (size_t)p.nchunks;
            const int batch = bh / (size_t)p.H;
            const int head = bh - (size_t)batch * p.H;
            const int group = head / p.heads_per_group;
            const int t0 = chunk_idx * chunk;
            const int p0 = ptile * out_bh;
            float *dec = smem;
            float *cb = dec + chunk;
            float *dtx = cb + chunk * chunk;
            float *cs = dtx + chunk * out_bh;
            float *hs = cs + 4 * chunk;

            for (int t = local; t < chunk; t += out_threads)
              dec[t] = p.decay[ssm2_dt_idx(batch, t0 + t, head, p.T, p.H)];
            for (int i = local; i < chunk * chunk / 4; i += out_threads) {
              const size_t base = cb_idx(p, batch, chunk_idx, group, 0, 0);
              st4(cb + (size_t)i * 4, ld4(p.cb + base + (size_t)i * 4));
            }
            // BF16 values are exact in TF32, so bf16(dt * x) is already the
            // operand.
            for (int i = local; i < chunk * out_bh / 4; i += out_threads) {
              const int elem = i * 4;
              const int t = elem / out_bh;
              const int lp = elem - t * out_bh;
              const float dtv = ssm2_from_bf16(
                  p.dt[ssm2_dt_idx(batch, t0 + t, head, p.T, p.H)]);
              float xv[4];
              ssm2_ld4_bf16(p.x + ssm2_x_idx(batch, t0 + t, head, p0 + lp, p.T,
                                             p.H, p.P),
                            xv);
              st4(dtx + elem, ssm2_round_bf16(dtv * xv[0]),
                  ssm2_round_bf16(dtv * xv[1]), ssm2_round_bf16(dtv * xv[2]),
                  ssm2_round_bf16(dtv * xv[3]));
            }

            float acc[4][4] = {};
            for (int n0 = 0; n0 < p.N; n0 += 4) {
              for (int t = local; t < chunk; t += out_threads)
                stage4(cs + t * 4, p.c + ssm2_bc_idx(batch, t0 + t, group, n0,
                                                     p.T, p.G, p.N));
              for (int lp = local; lp < out_bh; lp += out_threads)
                stage4(hs + lp * 4,
                       p.hiddens + ssm2_hid_idx(batch, head, chunk_idx, p0 + lp,
                                                n0, p.H, p.nchunks, p.P, p.N));
              sycl::group_barrier(work_group);
#pragma unroll
              for (int kk = 0; kk < 4; ++kk) {
                float creg[4], hreg[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                  creg[i] = cs[(row + i) * 4 + kk];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                  hreg[j] = hs[(col + j) * 4 + kk];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                  for (int j = 0; j < 4; ++j)
                    acc[i][j] += creg[i] * hreg[j];
              }
              sycl::group_barrier(work_group);
            }

            float er[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              er[i] = sycl::exp2(dec[row + i]);
#pragma unroll
              for (int j = 0; j < 4; ++j)
                acc[i][j] *= er[i];
            }
            for (int j = 0; j < chunk; ++j) {
              float breg[4];
#pragma unroll
              for (int jj = 0; jj < 4; ++jj)
                breg[jj] = dtx[j * out_bh + col + jj];
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                const float a =
                    j <= row + i
                        ? ssm2_round_tf32(cb[(row + i) * chunk + j] *
                                          sycl::exp2(dec[row + i] -
                                                             dec[j]))
                        : 0.f;
#pragma unroll
                for (int jj = 0; jj < 4; ++jj)
                  acc[i][jj] += a * breg[jj];
              }
            }
            const float skip = p.dskip[head];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const size_t idx = ssm2_x_idx(batch, t0 + row + i, head,
                                            p0 + col, p.T, p.H, p.P);
              float xv[4];
              ssm2_ld4_bf16(p.x + idx, xv);
              ssm2_st4_bf16(p.out + idx, acc[i][0] + skip * xv[0],
                            acc[i][1] + skip * xv[1], acc[i][2] + skip * xv[2],
                            acc[i][3] + skip * xv[3]);
            }
            sycl::group_barrier(work_group);
          }
        });
  });
}

// Tensor-core path: one subgroup owns each 16x16 output tile. NVIDIA maps the
// BF16 operations to mma.sync and the final interaction product to TF32 MMA.
// AMD gfx90a maps the BF16 operations to MFMA. Sg is the device wavefront,
// 32 or 64, and each work-group is that one subgroup.
template <int Sg>
static void launch_mma(sycl::queue &q, const Params &p, size_t max_groups) {
  const size_t state_blocks =
      (size_t)p.B * p.H * (p.P / 16) * (p.N / 16);
  const size_t state_groups = grid_groups(state_blocks, max_groups, Sg);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<ssm2_bf16, 1> weighted_b(p.chunk * 16, h);
    sycl::local_accessor<float, 1> state_tile(16 * 16, h);
    h.parallel_for(
        sycl::nd_range<1>(state_groups * (size_t)Sg, (size_t)Sg),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(Sg)]] {
#if SSM2_MMA_DEVICE
          const sycl::sub_group sg = item.get_sub_group();
          const int lane = item.get_local_id(0);
          const auto work_group = item.get_group();
          const size_t stride = work_group.get_group_range(0);
          auto *weighted =
              weighted_b.get_multi_ptr<sycl::access::decorated::no>().get();
          auto *tile =
              state_tile.get_multi_ptr<sycl::access::decorated::no>().get();
          for (size_t linear_block = work_group.get_group_id()[0];
               linear_block < state_blocks; linear_block += stride) {
            const int ntiles = p.N / 16;
            const int ptiles = p.P / 16;
            size_t block = linear_block;
            const int ntile = block % (size_t)ntiles;
            block /= (size_t)ntiles;
            const int ptile = block % (size_t)ptiles;
            block /= (size_t)ptiles;
            const int head = block % (size_t)p.H;
            const int batch = block / (size_t)p.H;
            const int group = head / p.heads_per_group;
            const int p0 = ptile * 16;
            const int n0 = ntile * 16;

            matrix::joint_matrix<sycl::sub_group, float,
                                 matrix::use::accumulator, 16, 16>
                state;
            matrix::joint_matrix_load(
                sg, state,
                matrix_global_ptr(
                    p.init_state +
                    ssm2_st_idx(batch, head, p0, n0, p.H, p.P, p.N)),
                p.N, matrix::layout::row_major);

            for (int chunk = 0; chunk < p.nchunks; ++chunk) {
              const int t0 = chunk * p.chunk;
              matrix::joint_matrix_store(sg, state, matrix_local_ptr(tile), 16,
                                         matrix::layout::row_major);
              sycl::group_barrier(work_group);
              for (int i = lane; i < 256; i += Sg) {
                const int pi = i / 16;
                const int ni = i - pi * 16;
                p.hiddens[ssm2_hid_idx(
                    batch, head, chunk, p0 + pi, n0 + ni, p.H, p.nchunks, p.P,
                    p.N)] = ssm2_to_bf16(tile[i]);
              }

              const float dlast = p.decay[ssm2_dt_idx(
                  batch, t0 + p.chunk - 1, head, p.T, p.H)];
              for (int i = lane; i < p.chunk * 16; i += Sg) {
                const int t = i / 16;
                const int n = i - t * 16;
                const float dec =
                    p.decay[ssm2_dt_idx(batch, t0 + t, head, p.T, p.H)];
                const float dtv = ssm2_from_bf16(
                    p.dt[ssm2_dt_idx(batch, t0 + t, head, p.T, p.H)]);
                const float bv = ssm2_from_bf16(p.b[ssm2_bc_idx(
                    batch, t0 + t, group, n0 + n, p.T, p.G, p.N)]);
                weighted[i] = ssm2_to_bf16(
                    bv * (dtv * sycl::exp2(dlast - dec)));
              }
              sycl::group_barrier(work_group);

              // Scale the carried state, still staged in tile, before adding
              // this chunk's BF16 MMA.
              const float scale = sycl::exp2(dlast);
              for (int i = lane; i < 256; i += Sg)
                tile[i] *= scale;
              sycl::group_barrier(work_group);
              matrix::joint_matrix_load(sg, state, matrix_local_ptr(tile), 16,
                                        matrix::layout::row_major);

              for (int k0 = 0; k0 < p.chunk; k0 += 16) {
                matrix::joint_matrix<sycl::sub_group, sycl_bfloat16,
                                     matrix::use::a, 16, 16, kMmaACol>
                    x_frag;
                matrix::joint_matrix<sycl::sub_group, sycl_bfloat16,
                                     matrix::use::b, 16, 16,
                                     matrix::layout::row_major>
                    b_frag;
                const auto *x_ptr =
                    reinterpret_cast<const sycl_bfloat16 *>(
                        p.x + ssm2_x_idx(batch, t0 + k0, head, p0, p.T, p.H,
                                        p.P));
                matrix::joint_matrix_load(sg, x_frag,
                                          matrix_global_ptr(x_ptr),
                                          p.H * p.P);
                matrix::joint_matrix_load(
                    sg, b_frag,
                    matrix_local_ptr(reinterpret_cast<sycl_bfloat16 *>(
                        weighted + k0 * 16)),
                    16);
                matrix::joint_matrix_mad(sg, state, x_frag, b_frag, state);
              }
              sycl::group_barrier(work_group);
            }
            matrix::joint_matrix_store(
                sg, state,
                matrix_global_ptr(
                    p.final_state +
                    ssm2_st_idx(batch, head, p0, n0, p.H, p.P, p.N)),
                p.N, matrix::layout::row_major);
            sycl::group_barrier(work_group);
          }
#else
          // Keep the host and device lambda captures identical.
          (void)item, (void)p, (void)state_blocks, (void)weighted_b,
              (void)state_tile;
#endif
        });
  });

  const size_t cb_blocks = (size_t)p.B * p.nchunks * p.G;
  const int cb_tiles = p.chunk / 16;
  const size_t cb_tile_blocks =
      cb_blocks * (size_t)cb_tiles * (size_t)cb_tiles;
  const size_t cb_groups = grid_groups(cb_tile_blocks, max_groups, Sg);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(cb_groups * (size_t)Sg, (size_t)Sg),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(Sg)]] {
#if SSM2_MMA_DEVICE
          const sycl::sub_group sg = item.get_sub_group();
          const auto work_group = item.get_group();
          const size_t stride = work_group.get_group_range(0);
          const int tiles = cb_tiles;
          const size_t tile_pairs = (size_t)tiles * (size_t)tiles;
          for (size_t linear = work_group.get_group_id()[0];
               linear < cb_tile_blocks; linear += stride) {
            const int tile = (int)(linear % tile_pairs);
            size_t block = linear / tile_pairs;
            const int group = block % (size_t)p.G;
            block /= (size_t)p.G;
            const int chunk_idx = block % (size_t)p.nchunks;
            const int batch = block / (size_t)p.nchunks;
            const int t0 = chunk_idx * p.chunk;
            {
              const int row = (tile / tiles) * 16;
              const int col = (tile % tiles) * 16;
              matrix::joint_matrix<sycl::sub_group, float,
                                   matrix::use::accumulator, 16, 16>
                  acc;
              matrix::joint_matrix_fill(sg, acc, 0.f);
              for (int n0 = 0; n0 < p.N; n0 += 16) {
                matrix::joint_matrix<sycl::sub_group, sycl_bfloat16,
                                     matrix::use::a, 16, 16, kMmaARow>
                    c_frag;
                matrix::joint_matrix<sycl::sub_group, sycl_bfloat16,
                                     matrix::use::b, 16, 16,
                                     matrix::layout::col_major>
                    b_frag;
                const auto *c_ptr =
                    reinterpret_cast<const sycl_bfloat16 *>(
                        p.c + ssm2_bc_idx(batch, t0 + row, group, n0, p.T, p.G,
                                         p.N));
                const auto *b_ptr =
                    reinterpret_cast<const sycl_bfloat16 *>(
                        p.b + ssm2_bc_idx(batch, t0 + col, group, n0, p.T, p.G,
                                         p.N));
                matrix::joint_matrix_load(sg, c_frag,
                                          matrix_global_ptr(c_ptr),
                                          p.G * p.N);
                matrix::joint_matrix_load(sg, b_frag,
                                          matrix_global_ptr(b_ptr),
                                          p.G * p.N);
                matrix::joint_matrix_mad(sg, acc, c_frag, b_frag, acc);
              }
              matrix::joint_matrix_store(
                  sg, acc,
                  matrix_global_ptr(
                      p.cb + cb_idx(p, batch, chunk_idx, group, row, col)),
                  p.chunk,
                  matrix::layout::row_major);
            }
          }
#else
          // The host pass retains the MMA body and captures these values,
          // whereas an Intel SPIR-V device image takes this scalar fallback.
          // Reference the same values so both passes agree on lambda layout.
          (void)item, (void)p, (void)cb_tiles, (void)cb_tile_blocks;
#endif
        });
  });

  const size_t out_blocks = (size_t)p.B * p.H * p.nchunks * (p.P / 16) *
                            (p.chunk / 16);
  const size_t out_groups = grid_groups(out_blocks, max_groups, Sg);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> sm(p.chunk * 16 + 256 + 128, h);
    h.parallel_for(
        sycl::nd_range<1>(out_groups * (size_t)Sg, (size_t)Sg),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(Sg)]] {
#if SSM2_MMA_DEVICE
          const sycl::sub_group sg = item.get_sub_group();
          const int lane = item.get_local_id(0);
          const auto work_group = item.get_group();
          const size_t stride = work_group.get_group_range(0);
          float *local =
              sm.get_multi_ptr<sycl::access::decorated::no>().get();
          float *dtx = local;
          float *acc_tile = dtx + p.chunk * 16;
          float *a_tile = acc_tile + 256;
          for (size_t linear_block = work_group.get_group_id()[0];
               linear_block < out_blocks; linear_block += stride) {
            size_t block = linear_block;
            const int row_tile = block % (size_t)(p.chunk / 16);
            block /= (size_t)(p.chunk / 16);
            const int ptile = block % (size_t)(p.P / 16);
            block /= (size_t)(p.P / 16);
            const int chunk_idx = block % (size_t)p.nchunks;
            const size_t bh = block / (size_t)p.nchunks;
            const int batch = bh / (size_t)p.H;
            const int head = bh - (size_t)batch * p.H;
            const int group = head / p.heads_per_group;
            const int t0 = chunk_idx * p.chunk;
            const int row0 = row_tile * 16;
            const int p0 = ptile * 16;

            for (int i = lane; i < p.chunk * 16; i += Sg) {
              const int t = i / 16;
              const int pi = i - t * 16;
              const float dtv = ssm2_from_bf16(
                  p.dt[ssm2_dt_idx(batch, t0 + t, head, p.T, p.H)]);
              const float xv = ssm2_from_bf16(p.x[ssm2_x_idx(
                  batch, t0 + t, head, p0 + pi, p.T, p.H, p.P)]);
              dtx[i] = ssm2_round_bf16(dtv * xv);
            }
            sycl::group_barrier(work_group);

            matrix::joint_matrix<sycl::sub_group, float,
                                 matrix::use::accumulator, 16, 16>
                acc;
            matrix::joint_matrix_fill(sg, acc, 0.f);
            for (int n0 = 0; n0 < p.N; n0 += 16) {
              matrix::joint_matrix<sycl::sub_group, sycl_bfloat16,
                                   matrix::use::a, 16, 16, kMmaARow>
                  c_frag;
              matrix::joint_matrix<sycl::sub_group, sycl_bfloat16,
                                   matrix::use::b, 16, 16,
                                   matrix::layout::col_major>
                  h_frag;
              const auto *c_ptr =
                  reinterpret_cast<const sycl_bfloat16 *>(
                      p.c + ssm2_bc_idx(batch, t0 + row0, group, n0, p.T, p.G,
                                       p.N));
              const auto *h_ptr =
                  reinterpret_cast<const sycl_bfloat16 *>(
                      p.hiddens + ssm2_hid_idx(batch, head, chunk_idx, p0, n0,
                                               p.H, p.nchunks, p.P, p.N));
              matrix::joint_matrix_load(sg, c_frag,
                                        matrix_global_ptr(c_ptr),
                                        p.G * p.N);
              matrix::joint_matrix_load(sg, h_frag,
                                        matrix_global_ptr(h_ptr), p.N);
              matrix::joint_matrix_mad(sg, acc, c_frag, h_frag, acc);
            }
            matrix::joint_matrix_store(sg, acc, matrix_local_ptr(acc_tile), 16,
                                       matrix::layout::row_major);
            sycl::group_barrier(work_group);
            for (int i = lane; i < 256; i += Sg) {
              const int row = i / 16;
              const float dec = p.decay[ssm2_dt_idx(
                  batch, t0 + row0 + row, head, p.T, p.H)];
              acc_tile[i] *= sycl::exp2(dec);
            }
            sycl::group_barrier(work_group);
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
            // AMD joint_matrix has BF16 16x16x16 MFMA and no TF32 16x8 shape.
            // Apply the same TF32-rounded product with scalar FMAs.
            for (int k0 = 0; k0 < p.chunk; k0 += 8) {
              for (int i = lane; i < 128; i += Sg) {
                const int row = i / 8;
                const int k = i - row * 8;
                const int j = k0 + k;
                float value = 0.f;
                if (j <= row0 + row) {
                  const float di = p.decay[ssm2_dt_idx(
                      batch, t0 + row0 + row, head, p.T, p.H)];
                  const float dj = p.decay[ssm2_dt_idx(
                      batch, t0 + j, head, p.T, p.H)];
                  value = ssm2_round_tf32(
                      p.cb[cb_idx(p, batch, chunk_idx, group, row0 + row, j)] *
                      sycl::exp2(di - dj));
                }
                a_tile[i] = value;
              }
              sycl::group_barrier(work_group);
              for (int i = lane; i < 256; i += Sg) {
                const int row = i / 16;
                const int pi = i - row * 16;
                float sum = acc_tile[i];
                const float *a_row = a_tile + row * 8;
                for (int k = 0; k < 8; ++k)
                  sum = sycl::fma(a_row[k], dtx[(k0 + k) * 16 + pi], sum);
                acc_tile[i] = sum;
              }
              sycl::group_barrier(work_group);
            }
#else
            matrix::joint_matrix_load(sg, acc, matrix_local_ptr(acc_tile), 16,
                                      matrix::layout::row_major);

            for (int k0 = 0; k0 < p.chunk; k0 += 8) {
              for (int i = lane; i < 128; i += Sg) {
                const int row = i / 8;
                const int k = i - row * 8;
                const int j = k0 + k;
                float value = 0.f;
                if (j <= row0 + row) {
                  const float di = p.decay[ssm2_dt_idx(
                      batch, t0 + row0 + row, head, p.T, p.H)];
                  const float dj = p.decay[ssm2_dt_idx(
                      batch, t0 + j, head, p.T, p.H)];
                  value = ssm2_round_tf32(
                      p.cb[cb_idx(p, batch, chunk_idx, group, row0 + row, j)] *
                      sycl::exp2(di - dj));
                }
                a_tile[i] = value;
              }
              sycl::group_barrier(work_group);
              matrix::joint_matrix<sycl::sub_group, matrix::precision::tf32,
                                   matrix::use::a, 16, 8,
                                   matrix::layout::row_major>
                  a_frag;
              matrix::joint_matrix<sycl::sub_group, matrix::precision::tf32,
                                   matrix::use::b, 8, 16,
                                   matrix::layout::row_major>
                  b_frag;
              matrix::joint_matrix_load(sg, a_frag,
                                        matrix_local_ptr(a_tile), 8);
              matrix::joint_matrix_load(
                  sg, b_frag, matrix_local_ptr(dtx + k0 * 16), 16);
              matrix::joint_matrix_mad(sg, acc, a_frag, b_frag, acc);
              sycl::group_barrier(work_group);
            }

            matrix::joint_matrix_store(sg, acc, matrix_local_ptr(acc_tile), 16,
                                       matrix::layout::row_major);
            sycl::group_barrier(work_group);
#endif
            const float skip = p.dskip[head];
            for (int i = lane; i < 256; i += Sg) {
              const int row = i / 16;
              const int pi = i - row * 16;
              const size_t idx = ssm2_x_idx(
                  batch, t0 + row0 + row, head, p0 + pi, p.T, p.H, p.P);
              p.out[idx] = ssm2_to_bf16(
                  acc_tile[i] + skip * ssm2_from_bf16(p.x[idx]));
            }
            sycl::group_barrier(work_group);
          }
#else
          (void)item, (void)p, (void)out_blocks, (void)sm;
#endif
        });
  });
}
