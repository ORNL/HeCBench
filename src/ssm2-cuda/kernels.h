#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#define SSM2_HAS_ROCWMMA 1
#else
#define SSM2_HAS_ROCWMMA 0
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_runtime.h>
#endif

#include <math.h>
#include <stddef.h>

#include "reference.h"

static constexpr float kLog2e = 1.4426950408889634f;
static constexpr int kThreads = 256;
static constexpr int kStatePTile = 16;
static constexpr int kStateNTile = 16;
static constexpr int kOutPTile = 8;

// Storage types follow ssm2_types.h: BF16 activations and chunk-entry states,
// FP32 parameters, decay, carried state, and the C @ B^T workspace.
struct SSM2Params {
  int B, T, H, G, P, N, chunk, nchunks, heads_per_group;
  const ssm2_bf16 *x;
  const float *a;
  const ssm2_bf16 *b;
  const ssm2_bf16 *c;
  const float *dskip;
  const ssm2_bf16 *dt;
  const float *init_state;
  float *decay;
  float *cb;
  ssm2_bf16 *hiddens;
  float *final_state;
  ssm2_bf16 *out;
};

__device__ __forceinline__ size_t cb_idx(const SSM2Params &p, int b, int chunk,
                                         int group, int i, int j);

#if defined(__HIPCC__)

namespace rw = rocwmma;

// gfx90a MFMA counterpart of the CUDA WMMA state kernel.
__device__ __forceinline__ void fwd_mfma_tile(const SSM2Params &p,
                                              size_t linear_block,
                                              ssm2_bf16 *weighted_b,
                                              float *state_tile) {
  const int ntiles = p.N / 16;
  const int ptiles = p.P / 16;
  size_t block = linear_block;
  const int ntile = block % (size_t)ntiles;
  block /= (size_t)ntiles;
  const int ptile = block % (size_t)ptiles;
  block /= (size_t)ptiles;
  const int h = block % (size_t)p.H;
  const int b = block / (size_t)p.H;
  const int group = h / p.heads_per_group;
  const int p0 = ptile * 16;
  const int n0 = ntile * 16;
  const int lane = threadIdx.x;

  rw::fragment<rw::accumulator, 16, 16, 16, float> state;
  rw::load_matrix_sync(
      state, p.init_state + ssm2_st_idx(b, h, p0, n0, p.H, p.P, p.N), p.N,
      rw::mem_row_major);

  for (int chunk = 0; chunk < p.nchunks; ++chunk) {
    const int t0 = chunk * p.chunk;
    rw::store_matrix_sync(state_tile, state, 16, rw::mem_row_major);
    __syncthreads();
    for (int i = lane; i < 256; i += warpSize) {
      const int pi = i / 16;
      const int ni = i - pi * 16;
      p.hiddens[ssm2_hid_idx(b, h, chunk, p0 + pi, n0 + ni, p.H, p.nchunks,
                             p.P, p.N)] = ssm2_to_bf16(state_tile[i]);
    }

    const float d_last =
        p.decay[ssm2_dt_idx(b, t0 + p.chunk - 1, h, p.T, p.H)];
    for (int i = lane; i < p.chunk * 16; i += warpSize) {
      const int t = i / 16;
      const int n = i - t * 16;
      const float dec = p.decay[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)];
      const float dtv =
          ssm2_from_bf16(p.dt[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)]);
      const float bv = ssm2_from_bf16(
          p.b[ssm2_bc_idx(b, t0 + t, group, n0 + n, p.T, p.G, p.N)]);
      weighted_b[i] = ssm2_to_bf16(bv * (dtv * exp2f(d_last - dec)));
    }
    __syncthreads();

    const float scale = exp2f(d_last);
    for (int i = lane; i < 256; i += warpSize)
      state_tile[i] *= scale;
    __syncthreads();
    rw::load_matrix_sync(state, state_tile, 16, rw::mem_row_major);

    for (int k0 = 0; k0 < p.chunk; k0 += 16) {
      rw::fragment<rw::matrix_a, 16, 16, 16, rw::bfloat16_t, rw::col_major>
          x_frag;
      rw::fragment<rw::matrix_b, 16, 16, 16, rw::bfloat16_t, rw::row_major>
          b_frag;
      const rw::bfloat16_t *x_ptr =
          reinterpret_cast<const rw::bfloat16_t *>(
              p.x + ssm2_x_idx(b, t0 + k0, h, p0, p.T, p.H, p.P));
      rw::load_matrix_sync(x_frag, x_ptr, p.H * p.P);
      rw::load_matrix_sync(
          b_frag,
          reinterpret_cast<const rw::bfloat16_t *>(weighted_b + k0 * 16), 16);
      rw::mma_sync(state, x_frag, b_frag, state);
    }
    __syncthreads();
  }
  rw::store_matrix_sync(
      p.final_state + ssm2_st_idx(b, h, p0, n0, p.H, p.P, p.N), state, p.N,
      rw::mem_row_major);
}


// Four waves cooperatively cover the 16x16 tiles in each C @ B^T matrix.
__device__ __forceinline__ void cb_mfma_tiles(const SSM2Params &p,
                                              size_t blocks) {
  const int wave = threadIdx.x / warpSize;
  const int waves = blockDim.x / warpSize;
  const int tiles = p.chunk / 16;
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x) {
    size_t block = linear_block;
    const int group = block % (size_t)p.G;
    block /= (size_t)p.G;
    const int chunk_idx = block % (size_t)p.nchunks;
    const int b = block / (size_t)p.nchunks;
    const int t0 = chunk_idx * p.chunk;
    for (int tile = wave; tile < tiles * tiles; tile += waves) {
      const int row = (tile / tiles) * 16;
      const int col = (tile % tiles) * 16;
      rw::fragment<rw::accumulator, 16, 16, 16, float> acc;
      rw::fill_fragment(acc, 0.f);
      for (int n0 = 0; n0 < p.N; n0 += 16) {
        rw::fragment<rw::matrix_a, 16, 16, 16, rw::bfloat16_t, rw::row_major>
            c_frag;
        rw::fragment<rw::matrix_b, 16, 16, 16, rw::bfloat16_t, rw::col_major>
            b_frag;
        const rw::bfloat16_t *c_ptr =
            reinterpret_cast<const rw::bfloat16_t *>(
                p.c + ssm2_bc_idx(b, t0 + row, group, n0, p.T, p.G, p.N));
        const rw::bfloat16_t *b_ptr =
            reinterpret_cast<const rw::bfloat16_t *>(
                p.b + ssm2_bc_idx(b, t0 + col, group, n0, p.T, p.G, p.N));
        rw::load_matrix_sync(c_frag, c_ptr, p.G * p.N);
        rw::load_matrix_sync(b_frag, b_ptr, p.G * p.N);
        rw::mma_sync(acc, c_frag, b_frag, acc);
      }
      rw::store_matrix_sync(
          p.cb + cb_idx(p, b, chunk_idx, group, row, col), acc, p.chunk,
          rw::mem_row_major);
    }
  }
}

// BF16 MFMA computes C @ H^T. The final product uses gfx90a FP32 MFMA on
// explicitly TF32-rounded operands, retaining the cuTile numerical contract.
__device__ __forceinline__ void out_mfma_tile(const SSM2Params &p,
                                              size_t linear_block, float *sm) {
  size_t block = linear_block;
  const int row_tile = block % (size_t)(p.chunk / 16);
  block /= (size_t)(p.chunk / 16);
  const int ptile = block % (size_t)(p.P / 16);
  block /= (size_t)(p.P / 16);
  const int chunk_idx = block % (size_t)p.nchunks;
  const size_t bh = block / (size_t)p.nchunks;
  const int b = bh / (size_t)p.H;
  const int h = bh - (size_t)b * p.H;
  const int group = h / p.heads_per_group;
  const int t0 = chunk_idx * p.chunk;
  const int row0 = row_tile * 16;
  const int p0 = ptile * 16;
  const int lane = threadIdx.x;

  float *dtx = sm;
  float *acc_tile = dtx + p.chunk * 16;
  float *a_tile = acc_tile + 256;

  for (int i = lane; i < p.chunk * 16; i += warpSize) {
    const int t = i / 16;
    const int pi = i - t * 16;
    const float dtv =
        ssm2_from_bf16(p.dt[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)]);
    const float xv = ssm2_from_bf16(
        p.x[ssm2_x_idx(b, t0 + t, h, p0 + pi, p.T, p.H, p.P)]);
    dtx[i] = ssm2_round_bf16(dtv * xv);
  }
  __syncthreads();

  rw::fragment<rw::accumulator, 16, 16, 16, float> acc;
  rw::fill_fragment(acc, 0.f);
  for (int n0 = 0; n0 < p.N; n0 += 16) {
    rw::fragment<rw::matrix_a, 16, 16, 16, rw::bfloat16_t, rw::row_major>
        c_frag;
    rw::fragment<rw::matrix_b, 16, 16, 16, rw::bfloat16_t, rw::col_major>
        h_frag;
    const rw::bfloat16_t *c_ptr = reinterpret_cast<const rw::bfloat16_t *>(
        p.c + ssm2_bc_idx(b, t0 + row0, group, n0, p.T, p.G, p.N));
    const rw::bfloat16_t *h_ptr = reinterpret_cast<const rw::bfloat16_t *>(
        p.hiddens +
        ssm2_hid_idx(b, h, chunk_idx, p0, n0, p.H, p.nchunks, p.P, p.N));
    rw::load_matrix_sync(c_frag, c_ptr, p.G * p.N);
    rw::load_matrix_sync(h_frag, h_ptr, p.N);
    rw::mma_sync(acc, c_frag, h_frag, acc);
  }
  rw::store_matrix_sync(acc_tile, acc, 16, rw::mem_row_major);
  __syncthreads();
  for (int i = lane; i < 256; i += warpSize) {
    const int row = i / 16;
    const float dec =
        p.decay[ssm2_dt_idx(b, t0 + row0 + row, h, p.T, p.H)];
    acc_tile[i] *= exp2f(dec);
  }
  __syncthreads();

  // gfx90a f32 MFMA is 16x16x4.
  rw::fragment<rw::accumulator, 16, 16, 4, float> acc_f32;
  rw::load_matrix_sync(acc_f32, acc_tile, 16, rw::mem_row_major);
  for (int k0 = 0; k0 < p.chunk; k0 += 4) {
    for (int i = lane; i < 16 * 4; i += warpSize) {
      const int row = i / 4;
      const int k = i - row * 4;
      const int j = k0 + k;
      float value = 0.f;
      if (j <= row0 + row) {
        const float di =
            p.decay[ssm2_dt_idx(b, t0 + row0 + row, h, p.T, p.H)];
        const float dj = p.decay[ssm2_dt_idx(b, t0 + j, h, p.T, p.H)];
        value = ssm2_round_tf32(
            p.cb[cb_idx(p, b, chunk_idx, group, row0 + row, j)] *
            exp2f(di - dj));
      }
      a_tile[i] = value;
    }
    __syncthreads();
    rw::fragment<rw::matrix_a, 16, 16, 4, float, rw::row_major> a_frag;
    rw::fragment<rw::matrix_b, 16, 16, 4, float, rw::row_major> b_frag;
    rw::load_matrix_sync(a_frag, a_tile, 4);
    rw::load_matrix_sync(b_frag, dtx + k0 * 16, 16);
    rw::mma_sync(acc_f32, a_frag, b_frag, acc_f32);
    __syncthreads();
  }

  rw::store_matrix_sync(acc_tile, acc_f32, 16, rw::mem_row_major);
  __syncthreads();
  const float skip = p.dskip[h];
  for (int i = lane; i < 256; i += warpSize) {
    const int row = i / 16;
    const int pi = i - row * 16;
    const size_t idx =
        ssm2_x_idx(b, t0 + row0 + row, h, p0 + pi, p.T, p.H, p.P);
    p.out[idx] =
        ssm2_to_bf16(acc_tile[i] + skip * ssm2_from_bf16(p.x[idx]));
  }
  __syncthreads();
}

__global__ void fwd_mfma_kernel(SSM2Params p, size_t blocks) {
  extern __shared__ unsigned char storage[];
  ssm2_bf16 *weighted_b = reinterpret_cast<ssm2_bf16 *>(storage);
  float *state_tile =
      reinterpret_cast<float *>(weighted_b + (size_t)p.chunk * 16);
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x)
    fwd_mfma_tile(p, linear_block, weighted_b, state_tile);
}

__global__ void cb_mfma_kernel(SSM2Params p, size_t blocks) {
  cb_mfma_tiles(p, blocks);
}

__global__ void out_mfma_kernel(SSM2Params p, size_t blocks) {
  extern __shared__ float sm[];
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x)
    out_mfma_tile(p, linear_block, sm);
}

#endif

// Flatten an element of the per-batch, per-chunk grouped C @ B^T matrix.
__device__ __forceinline__ size_t cb_idx(const SSM2Params &p, int b, int chunk,
                                         int group, int i, int j) {
  return (((size_t)b * (size_t)p.nchunks + (size_t)chunk) * (size_t)p.G +
          (size_t)group) *
             (size_t)p.chunk * (size_t)p.chunk +
         (size_t)i * (size_t)p.chunk + (size_t)j;
}

// Compute each head's inclusive chunk-local cumulative decay.
__global__ void chunk_cumsum_kernel(SSM2Params p, size_t blocks) {
  const int local = threadIdx.x;
  if (local >= p.chunk)
    return;

  extern __shared__ float sm[];
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x) {
    size_t block = linear_block;
    const int h = block % (size_t)p.H;
    block /= (size_t)p.H;
    const int chunk_idx = block % (size_t)p.nchunks;
    const int b = block / (size_t)p.nchunks;
    const int t = chunk_idx * p.chunk + local;
    sm[local] = p.a[h] * ssm2_from_bf16(p.dt[ssm2_dt_idx(b, t, h, p.T, p.H)]);
    __syncthreads();

    for (int offset = 1; offset < p.chunk; offset <<= 1) {
      float add = 0.f;
      if (local >= offset)
        add = sm[local - offset];
      __syncthreads();
      sm[local] += add;
      __syncthreads();
    }
    p.decay[ssm2_dt_idx(b, t, h, p.T, p.H)] = sm[local] * kLog2e;
    __syncthreads();
  }
}

// Advance batch/head states directly from global memory using a grid stride.
__global__ void fwd_pass_state_global_kernel(SSM2Params p, size_t blocks) {
  for (size_t bh = blockIdx.x; bh < blocks; bh += gridDim.x) {
    const int b = bh / (size_t)p.H;
    const int h = bh - (size_t)b * p.H;
    const int group = h / p.heads_per_group;
    const int tid = threadIdx.x;
    const size_t PN = (size_t)p.P * (size_t)p.N;

    for (size_t i = tid; i < PN; i += blockDim.x) {
      const int head = (int)(i / (size_t)p.N);
      const int n = (int)(i - (size_t)head * (size_t)p.N);
      p.final_state[ssm2_st_idx(b, h, head, n, p.H, p.P, p.N)] =
          p.init_state[ssm2_st_idx(b, h, head, n, p.H, p.P, p.N)];
    }
    __syncthreads();

    for (int chunk = 0; chunk < p.nchunks; ++chunk) {
      const int t0 = chunk * p.chunk;
      for (size_t i = tid; i < PN; i += blockDim.x) {
        const int head = (int)(i / (size_t)p.N);
        const int n = (int)(i - (size_t)head * (size_t)p.N);
        p.hiddens[ssm2_hid_idx(b, h, chunk, head, n, p.H, p.nchunks, p.P,
                               p.N)] =
            ssm2_to_bf16(p.final_state[ssm2_st_idx(b, h, head, n, p.H, p.P,
                                                   p.N)]);
      }
      __syncthreads();

      const float d_last =
          p.decay[ssm2_dt_idx(b, t0 + p.chunk - 1, h, p.T, p.H)];
      const float scale = exp2f(d_last);
      for (size_t i = tid; i < PN; i += blockDim.x) {
        const int head = (int)(i / (size_t)p.N);
        const int n = (int)(i - (size_t)head * (size_t)p.N);
        float acc =
            scale * p.final_state[ssm2_st_idx(b, h, head, n, p.H, p.P, p.N)];
        for (int t = 0; t < p.chunk; ++t) {
          const int gt = t0 + t;
          const float dec = p.decay[ssm2_dt_idx(b, gt, h, p.T, p.H)];
          const float dtv = ssm2_from_bf16(p.dt[ssm2_dt_idx(b, gt, h, p.T, p.H)]);
          const float xv =
              ssm2_from_bf16(p.x[ssm2_x_idx(b, gt, h, head, p.T, p.H, p.P)]);
          const float bv =
              ssm2_from_bf16(p.b[ssm2_bc_idx(b, gt, group, n, p.T, p.G, p.N)]);
          acc += xv * ssm2_round_bf16(bv * (dtv * exp2f(d_last - dec)));
        }
        p.final_state[ssm2_st_idx(b, h, head, n, p.H, p.P, p.N)] = acc;
      }
      __syncthreads();
    }
  }
}

// Advance tiled P x N state planes so small B*H shapes expose more parallelism.
__global__ void fwd_pass_state_tiled_kernel(SSM2Params p, size_t blocks) {
  const int ptiles = (p.P + kStatePTile - 1) / kStatePTile;
  const int ntiles = (p.N + kStateNTile - 1) / kStateNTile;
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x) {
    size_t block = linear_block;
    const int ntile = block % (size_t)ntiles;
    block /= (size_t)ntiles;
    const int ptile = block % (size_t)ptiles;
    block /= (size_t)ptiles;
    const int h = block % (size_t)p.H;
    const int b = block / (size_t)p.H;
    const int group = h / p.heads_per_group;
    const int p0 = ptile * kStatePTile;
    const int n0 = ntile * kStateNTile;
    const int tid = threadIdx.x;

    extern __shared__ float sm[];
    float *x_s = sm;
    float *b_s = x_s + p.chunk * kStatePTile;

    const int lp = tid / kStateNTile;
    const int ln = tid % kStateNTile;
    const int hp = p0 + lp;
    const int sn = n0 + ln;
    const bool valid = hp < p.P && sn < p.N;
    if (valid) {
      p.final_state[ssm2_st_idx(b, h, hp, sn, p.H, p.P, p.N)] =
          p.init_state[ssm2_st_idx(b, h, hp, sn, p.H, p.P, p.N)];
    }

    for (int chunk = 0; chunk < p.nchunks; ++chunk) {
      const int t0 = chunk * p.chunk;
      if (valid) {
        p.hiddens[ssm2_hid_idx(b, h, chunk, hp, sn, p.H, p.nchunks, p.P, p.N)] =
            ssm2_to_bf16(p.final_state[ssm2_st_idx(b, h, hp, sn, p.H, p.P, p.N)]);
      }
      for (int i = tid; i < p.chunk * kStatePTile; i += blockDim.x) {
        const int t = i / kStatePTile;
        const int pp = p0 + i % kStatePTile;
        x_s[i] = pp < p.P ? ssm2_from_bf16(p.x[ssm2_x_idx(b, t0 + t, h, pp, p.T,
                                                          p.H, p.P)])
                          : 0.f;
      }
      const float d_last =
          p.decay[ssm2_dt_idx(b, t0 + p.chunk - 1, h, p.T, p.H)];
      // Stage B already weighted by dt * exp2(d_last - d) and rounded to BF16.
      for (int i = tid; i < p.chunk * kStateNTile; i += blockDim.x) {
        const int t = i / kStateNTile;
        const int nn = n0 + i % kStateNTile;
        float bw = 0.f;
        if (nn < p.N) {
          const float dec = p.decay[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)];
          const float dtv =
              ssm2_from_bf16(p.dt[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)]);
          const float bv = ssm2_from_bf16(
              p.b[ssm2_bc_idx(b, t0 + t, group, nn, p.T, p.G, p.N)]);
          bw = ssm2_round_bf16(bv * (dtv * exp2f(d_last - dec)));
        }
        b_s[i] = bw;
      }
      __syncthreads();
      if (valid) {
        float acc = exp2f(d_last) *
                    p.final_state[ssm2_st_idx(b, h, hp, sn, p.H, p.P, p.N)];
#pragma unroll 4
        for (int t = 0; t < p.chunk; ++t)
          acc += x_s[t * kStatePTile + lp] * b_s[t * kStateNTile + ln];
        p.final_state[ssm2_st_idx(b, h, hp, sn, p.H, p.P, p.N)] = acc;
      }
      __syncthreads();
    }
  }
}

// Materialize grouped C @ B^T matrices for all batches and chunks.
__global__ void compute_cb_kernel(SSM2Params p, size_t blocks) {
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x) {
    size_t block = linear_block;
    const int group = block % (size_t)p.G;
    block /= (size_t)p.G;
    const int chunk = block % (size_t)p.nchunks;
    const int b = block / (size_t)p.nchunks;
    const int t0 = chunk * p.chunk;
    for (int ij = threadIdx.x; ij < p.chunk * p.chunk; ij += blockDim.x) {
      const int i = ij / p.chunk;
      const int j = ij - i * p.chunk;
      float dot = 0.f;
      for (int n = 0; n < p.N; ++n)
        dot += ssm2_from_bf16(p.c[ssm2_bc_idx(b, t0 + i, group, n, p.T, p.G,
                                              p.N)]) *
               ssm2_from_bf16(p.b[ssm2_bc_idx(b, t0 + j, group, n, p.T, p.G,
                                              p.N)]);
      p.cb[cb_idx(p, b, chunk, group, i, j)] = dot;
    }
  }
}

// Produce output for one head-dimension tile using shared input and C @ B^T.
__global__ void compute_out_tiled_kernel(SSM2Params p, size_t blocks) {
  const int ptiles = (p.P + kOutPTile - 1) / kOutPTile;
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x) {
    size_t block = linear_block;
    const int ptile = block % (size_t)ptiles;
    block /= (size_t)ptiles;
    const int h = block % (size_t)p.H;
    block /= (size_t)p.H;
    const int chunk = block % (size_t)p.nchunks;
    const int b = block / (size_t)p.nchunks;
    const int group = h / p.heads_per_group;
    const int p0 = ptile * kOutPTile;
    const int t0 = chunk * p.chunk;
    const int tid = threadIdx.x;

    extern __shared__ float sm[];
    float *dec_s = sm;
    float *dtx_s = dec_s + p.chunk;
    float *cb_s = dtx_s + p.chunk * kOutPTile;
    for (int t = tid; t < p.chunk; t += blockDim.x)
      dec_s[t] = p.decay[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)];
    // BF16 values are exact in TF32, so bf16(dt * x) is already the operand.
    for (int i = tid; i < p.chunk * kOutPTile; i += blockDim.x) {
      const int t = i / kOutPTile;
      const int hp = p0 + i % kOutPTile;
      float dtx = 0.f;
      if (hp < p.P)
        dtx = ssm2_round_bf16(
            ssm2_from_bf16(p.dt[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)]) *
            ssm2_from_bf16(p.x[ssm2_x_idx(b, t0 + t, h, hp, p.T, p.H, p.P)]));
      dtx_s[i] = dtx;
    }
    for (int ij = tid; ij < p.chunk * p.chunk; ij += blockDim.x) {
      const int i = ij / p.chunk;
      const int j = ij - i * p.chunk;
      cb_s[ij] = p.cb[cb_idx(p, b, chunk, group, i, j)];
    }
    __syncthreads();

    const float skip = p.dskip[h];
    for (int ip = tid; ip < p.chunk * kOutPTile; ip += blockDim.x) {
      const int t = ip / kOutPTile;
      const int lp = ip % kOutPTile;
      const int hp = p0 + lp;
      if (hp >= p.P)
        continue;
      float carried = 0.f;
      for (int n = 0; n < p.N; ++n)
        carried += ssm2_from_bf16(p.c[ssm2_bc_idx(b, t0 + t, group, n, p.T, p.G,
                                                  p.N)]) *
                   ssm2_from_bf16(p.hiddens[ssm2_hid_idx(
                       b, h, chunk, hp, n, p.H, p.nchunks, p.P, p.N)]);
      float acc = carried * exp2f(dec_s[t]);
      for (int j = 0; j <= t; ++j)
        acc += ssm2_round_tf32(cb_s[t * p.chunk + j] *
                               exp2f(dec_s[t] - dec_s[j])) *
               dtx_s[j * kOutPTile + lp];
      acc += skip *
             ssm2_from_bf16(p.x[ssm2_x_idx(b, t0 + t, h, hp, p.T, p.H, p.P)]);
      p.out[ssm2_x_idx(b, t0 + t, h, hp, p.T, p.H, p.P)] = ssm2_to_bf16(acc);
    }
    __syncthreads();
  }
}

// Produce output directly from global memory using a grid stride.
__global__ void compute_out_global_kernel(SSM2Params p, size_t blocks) {
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x) {
    size_t block = linear_block;
    const int h = block % (size_t)p.H;
    block /= (size_t)p.H;
    const int chunk = block % (size_t)p.nchunks;
    const int b = block / (size_t)p.nchunks;
    const int group = h / p.heads_per_group;
    const int tid = threadIdx.x;
    const int t0 = chunk * p.chunk;
    const float skip = p.dskip[h];

    for (int ip = tid; ip < p.chunk * p.P; ip += blockDim.x) {
      const int t = ip / p.P;
      const int head = ip - t * p.P;
      const int gt = t0 + t;
      const float dec_t = p.decay[ssm2_dt_idx(b, gt, h, p.T, p.H)];
      const float xv =
          ssm2_from_bf16(p.x[ssm2_x_idx(b, gt, h, head, p.T, p.H, p.P)]);
      float carried = 0.f;
      for (int n = 0; n < p.N; ++n) {
        const float cv =
            ssm2_from_bf16(p.c[ssm2_bc_idx(b, gt, group, n, p.T, p.G, p.N)]);
        const float hs = ssm2_from_bf16(p.hiddens[ssm2_hid_idx(
            b, h, chunk, head, n, p.H, p.nchunks, p.P, p.N)]);
        carried += cv * hs;
      }
      float acc = carried * exp2f(dec_t);
      for (int j = 0; j <= t; ++j) {
        const int gj = t0 + j;
        float dot = 0.f;
        for (int n = 0; n < p.N; ++n) {
          dot += ssm2_from_bf16(
                     p.c[ssm2_bc_idx(b, gt, group, n, p.T, p.G, p.N)]) *
                 ssm2_from_bf16(
                     p.b[ssm2_bc_idx(b, gj, group, n, p.T, p.G, p.N)]);
        }
        const float dec_j = p.decay[ssm2_dt_idx(b, gj, h, p.T, p.H)];
        const float dtv = ssm2_from_bf16(p.dt[ssm2_dt_idx(b, gj, h, p.T, p.H)]);
        const float xj =
            ssm2_from_bf16(p.x[ssm2_x_idx(b, gj, h, head, p.T, p.H, p.P)]);
        acc += ssm2_round_tf32(dot * exp2f(dec_t - dec_j)) *
               ssm2_round_bf16(dtv * xj);
      }
      acc += skip * xv;
      p.out[ssm2_x_idx(b, gt, h, head, p.T, p.H, p.P)] = ssm2_to_bf16(acc);
    }
  }
}

// Load four contiguous floats through one aligned vector transaction.
__device__ __forceinline__ float4 ld4(const float *p) {
  return *reinterpret_cast<const float4 *>(p);
}

// Store four contiguous floats through one aligned vector transaction.
__device__ __forceinline__ void st4(float *p, float4 v) {
  *reinterpret_cast<float4 *>(p) = v;
}

__device__ __forceinline__ void st4(float *p, float a, float b, float c,
                                    float d) {
  st4(p, make_float4(a, b, c, d));
}

// Widen four BF16 values from global memory into one float4 shared slot.
__device__ __forceinline__ void stage4(float *dst, const ssm2_bf16 *src) {
  float v[4];
  ssm2_ld4_bf16(src, v);
  st4(dst, v[0], v[1], v[2], v[3]);
}

// Compute raw C @ B^T tiles; head-specific decay masks are applied later.
__global__ void cb_gemm_kernel(SSM2Params p, size_t blocks) {
  const int chunk = p.chunk;
  const int tiles = chunk >> 2;
  const int tid = threadIdx.x;
  const int tile_r = tid / tiles;
  const int tile_c = tid - tile_r * tiles;
  const int row = tile_r << 2;
  const int col = tile_c << 2;
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x) {
    size_t block = linear_block;
    const int group = block % (size_t)p.G;
    block /= (size_t)p.G;
    const int chunk_idx = block % (size_t)p.nchunks;
    const int b = block / (size_t)p.nchunks;
    const int t0 = chunk_idx * chunk;

    float acc[4][4] = {};
    extern __shared__ float sm[];
    float *cs = sm;
    float *bs = cs + 4 * chunk;

    for (int n0 = 0; n0 < p.N; n0 += 4) {
      for (int t = tid; t < chunk; t += blockDim.x) {
        stage4(cs + t * 4, p.c + ssm2_bc_idx(b, t0 + t, group, n0, p.T, p.G, p.N));
        stage4(bs + t * 4, p.b + ssm2_bc_idx(b, t0 + t, group, n0, p.T, p.G, p.N));
      }
      __syncthreads();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        float creg[4], breg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          creg[i] = cs[(row + i) * 4 + kk];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          breg[j] = bs[(col + j) * 4 + kk];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            acc[i][j] += creg[i] * breg[j];
      }
      __syncthreads();
    }

    float *dst = p.cb + cb_idx(p, b, chunk_idx, group, row, col);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      st4(dst + i * chunk, acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    __syncthreads();
  }
}

// Advance tiled state matrices with a register-blocked X^T @ weighted-B GEMM.
__global__ void fwd_gemm_kernel(SSM2Params p, int BH, int BS, size_t blocks) {
  const int tiles_c = BS >> 2;
  const int tile_threads = (BH >> 2) * tiles_c;
  const int splits = blockDim.x / tile_threads;
  const int split = threadIdx.x / tile_threads;
  const int tid = threadIdx.x - split * tile_threads;
  const int tile_r = tid / tiles_c;
  const int tile_c = tid - tile_r * tiles_c;
  const int row = tile_r << 2;
  const int col = tile_c << 2;
  const int ntiles = p.N / BS;
  const int ptiles = p.P / BH;
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x) {
    size_t block = linear_block;
    const int ntile = block % (size_t)ntiles;
    block /= (size_t)ntiles;
    const int ptile = block % (size_t)ptiles;
    block /= (size_t)ptiles;
    const int h = block % (size_t)p.H;
    const int b = block / (size_t)p.H;
    const int group = h / p.heads_per_group;
    const int pbase = ptile * BH;
    const int nbase = ntile * BS;
    const int prow = pbase + row;
    const int ncol = nbase + col;

    float hid[4][4] = {};
    if (split == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float4 v = ld4(p.init_state +
                             ssm2_st_idx(b, h, prow + i, ncol, p.H, p.P, p.N));
        hid[i][0] = v.x;
        hid[i][1] = v.y;
        hid[i][2] = v.z;
        hid[i][3] = v.w;
      }
    }

    extern __shared__ float sm[];
    float *xs = sm;
    float *bs = xs + p.chunk * BH;
    float *parts = bs + p.chunk * BS;

    for (int chunk = 0; chunk < p.nchunks; ++chunk) {
      const int t0 = chunk * p.chunk;
      if (split == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
          ssm2_st4_bf16(p.hiddens + ssm2_hid_idx(b, h, chunk, prow + i, ncol,
                                                 p.H, p.nchunks, p.P, p.N),
                        hid[i][0], hid[i][1], hid[i][2], hid[i][3]);
      }

      for (int i = threadIdx.x; i < p.chunk * BH / 4; i += blockDim.x) {
        const int elem = i * 4;
        const int t = elem / BH;
        const int lp = elem - t * BH;
        stage4(xs + elem,
               p.x + ssm2_x_idx(b, t0 + t, h, pbase + lp, p.T, p.H, p.P));
      }
      const float d_last =
          p.decay[ssm2_dt_idx(b, t0 + p.chunk - 1, h, p.T, p.H)];
      // Stage B already weighted by dt * exp2(d_last - d) and rounded to BF16.
      for (int i = threadIdx.x; i < p.chunk * BS / 4; i += blockDim.x) {
        const int elem = i * 4;
        const int t = elem / BS;
        const int ln = elem - t * BS;
        const float dec = p.decay[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)];
        const float dtv =
            ssm2_from_bf16(p.dt[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)]);
        const float w = dtv * exp2f(d_last - dec);
        float bv[4];
        ssm2_ld4_bf16(
            p.b + ssm2_bc_idx(b, t0 + t, group, nbase + ln, p.T, p.G, p.N), bv);
        st4(bs + elem, ssm2_round_bf16(bv[0] * w), ssm2_round_bf16(bv[1] * w),
            ssm2_round_bf16(bv[2] * w), ssm2_round_bf16(bv[3] * w));
      }
      __syncthreads();

      float part[4][4] = {};
      const int t_lo = split * (p.chunk / splits);
      const int t_hi = t_lo + p.chunk / splits;
      for (int t = t_lo; t < t_hi; ++t) {
        float xv[4], bv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
          xv[i] = xs[t * BH + row + i];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          bv[j] = bs[t * BS + col + j];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            part[i][j] += xv[i] * bv[j];
      }
      float *mine = parts + (split * tile_threads + tid) * 16;
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          mine[i * 4 + j] = part[i][j];
      __syncthreads();

      if (split == 0) {
        const float scale = exp2f(d_last);
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            hid[i][j] *= scale;
        for (int s = 0; s < splits; ++s) {
          const float *src = parts + (s * tile_threads + tid) * 16;
#pragma unroll
          for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
              hid[i][j] += src[i * 4 + j];
        }
      }
      __syncthreads();
    }

    if (split == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        st4(p.final_state + ssm2_st_idx(b, h, prow + i, ncol, p.H, p.P, p.N),
            hid[i][0], hid[i][1], hid[i][2], hid[i][3]);
    }
    __syncthreads();
  }
}

// Combine carried state, causal C @ B^T, and skip terms into tiled output.
__global__ void compute_out_gemm_kernel(SSM2Params p, int BH, size_t blocks) {
  const int chunk = p.chunk;
  const int tiles_c = BH >> 2;
  const int tid = threadIdx.x;
  const int tile_r = tid / tiles_c;
  const int tile_c = tid - tile_r * tiles_c;
  const int row = tile_r << 2;
  const int col = tile_c << 2;
  const int ptiles = p.P / BH;
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x) {
    size_t block = linear_block;
    const int ptile = block % (size_t)ptiles;
    block /= (size_t)ptiles;
    const int chunk_idx = block % (size_t)p.nchunks;
    const size_t bh = block / (size_t)p.nchunks;
    const int p0 = ptile * BH;
    const int b = bh / (size_t)p.H;
    const int h = bh - (size_t)b * p.H;
    const int group = h / p.heads_per_group;
    const int t0 = chunk_idx * chunk;

    extern __shared__ float sm[];
    float *dec = sm;
    float *cb = dec + chunk;
    float *dtx = cb + chunk * chunk;
    float *cs = dtx + chunk * BH;
    float *hs = cs + 4 * chunk;

    for (int t = tid; t < chunk; t += blockDim.x)
      dec[t] = p.decay[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)];
    for (int i = tid; i < chunk * chunk / 4; i += blockDim.x) {
      const size_t base = cb_idx(p, b, chunk_idx, group, 0, 0);
      st4(cb + i * 4, ld4(p.cb + base + i * 4));
    }
    // BF16 values are exact in TF32, so bf16(dt * x) is already the operand.
    for (int i = tid; i < chunk * BH / 4; i += blockDim.x) {
      const int elem = i * 4;
      const int t = elem / BH;
      const int lp = elem - t * BH;
      const float dtv = ssm2_from_bf16(p.dt[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)]);
      float xv[4];
      ssm2_ld4_bf16(p.x + ssm2_x_idx(b, t0 + t, h, p0 + lp, p.T, p.H, p.P), xv);
      st4(dtx + elem, ssm2_round_bf16(dtv * xv[0]), ssm2_round_bf16(dtv * xv[1]),
          ssm2_round_bf16(dtv * xv[2]), ssm2_round_bf16(dtv * xv[3]));
    }

    float acc[4][4] = {};
    for (int n0 = 0; n0 < p.N; n0 += 4) {
      for (int t = tid; t < chunk; t += blockDim.x)
        stage4(cs + t * 4, p.c + ssm2_bc_idx(b, t0 + t, group, n0, p.T, p.G, p.N));
      for (int lp = tid; lp < BH; lp += blockDim.x)
        stage4(hs + lp * 4, p.hiddens + ssm2_hid_idx(b, h, chunk_idx, p0 + lp,
                                                     n0, p.H, p.nchunks, p.P,
                                                     p.N));
      __syncthreads();
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
      __syncthreads();
    }

    float er[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      er[i] = exp2f(dec[row + i]);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[i][j] *= er[i];
    }

    for (int j = 0; j < chunk; ++j) {
      float breg[4];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
        breg[jj] = dtx[j * BH + col + jj];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float a = (j <= row + i)
                            ? ssm2_round_tf32(cb[(row + i) * chunk + j] *
                                              exp2f(dec[row + i] - dec[j]))
                            : 0.f;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
          acc[i][jj] += a * breg[jj];
      }
    }

    const float skip = p.dskip[h];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const size_t idx = ssm2_x_idx(b, t0 + row + i, h, p0 + col, p.T, p.H, p.P);
      float xv[4];
      ssm2_ld4_bf16(p.x + idx, xv);
      ssm2_st4_bf16(p.out + idx, acc[i][0] + skip * xv[0],
                    acc[i][1] + skip * xv[1], acc[i][2] + skip * xv[2],
                    acc[i][3] + skip * xv[3]);
    }
    __syncthreads();
  }
}

#if !defined(__HIPCC__)

namespace wmma = nvcuda::wmma;

// One warp owns one 16x16 P-by-N state tile. X is loaded as a column-major
// [P, chunk] matrix directly from [chunk, P]. Weighted B is rounded to BF16
// exactly where the cuTile kernel casts it, then staged row-major.
__device__ __forceinline__ void fwd_wmma_tile(const SSM2Params &p,
                                              size_t linear_block,
                                              ssm2_bf16 *weighted_b,
                                              float *state_tile) {
  const int ntiles = p.N / 16;
  const int ptiles = p.P / 16;
  size_t block = linear_block;
  const int ntile = block % (size_t)ntiles;
  block /= (size_t)ntiles;
  const int ptile = block % (size_t)ptiles;
  block /= (size_t)ptiles;
  const int h = block % (size_t)p.H;
  const int b = block / (size_t)p.H;
  const int group = h / p.heads_per_group;
  const int p0 = ptile * 16;
  const int n0 = ntile * 16;
  const int lane = threadIdx.x;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> state;
  wmma::load_matrix_sync(
      state, p.init_state + ssm2_st_idx(b, h, p0, n0, p.H, p.P, p.N), p.N,
      wmma::mem_row_major);

  for (int chunk = 0; chunk < p.nchunks; ++chunk) {
    const int t0 = chunk * p.chunk;

    wmma::store_matrix_sync(state_tile, state, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
      const int pi = i / 16;
      const int ni = i - pi * 16;
      p.hiddens[ssm2_hid_idx(b, h, chunk, p0 + pi, n0 + ni, p.H,
                             p.nchunks, p.P, p.N)] =
          ssm2_to_bf16(state_tile[i]);
    }

    const float d_last =
        p.decay[ssm2_dt_idx(b, t0 + p.chunk - 1, h, p.T, p.H)];
    for (int i = lane; i < p.chunk * 16; i += 32) {
      const int t = i / 16;
      const int n = i - t * 16;
      const float dec = p.decay[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)];
      const float dtv =
          ssm2_from_bf16(p.dt[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)]);
      const float bv = ssm2_from_bf16(
          p.b[ssm2_bc_idx(b, t0 + t, group, n0 + n, p.T, p.G, p.N)]);
      weighted_b[i] = ssm2_to_bf16(bv * (dtv * exp2f(d_last - dec)));
    }
    __syncwarp();

    // Scale the carried state, still staged in state_tile, before adding this
    // chunk's BF16 MMA.
    const float scale = exp2f(d_last);
    for (int i = lane; i < 256; i += 32)
      state_tile[i] *= scale;
    __syncwarp();
    wmma::load_matrix_sync(state, state_tile, 16, wmma::mem_row_major);

    for (int k0 = 0; k0 < p.chunk; k0 += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::col_major>
          x_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          b_frag;
      const __nv_bfloat16 *x_ptr =
          reinterpret_cast<const __nv_bfloat16 *>(
              p.x + ssm2_x_idx(b, t0 + k0, h, p0, p.T, p.H, p.P));
      wmma::load_matrix_sync(x_frag, x_ptr, p.H * p.P);
      wmma::load_matrix_sync(
          b_frag,
          reinterpret_cast<const __nv_bfloat16 *>(weighted_b + k0 * 16), 16);
      wmma::mma_sync(state, x_frag, b_frag, state);
    }
    __syncwarp();
  }

  wmma::store_matrix_sync(
      p.final_state + ssm2_st_idx(b, h, p0, n0, p.H, p.P, p.N), state, p.N,
      wmma::mem_row_major);
}

// Each warp computes one 16x16 C @ B^T tile directly from BF16 global memory.
__device__ __forceinline__ void cb_wmma_tiles(const SSM2Params &p,
                                              size_t blocks) {
  const int warp = threadIdx.x / 32;
  const int warps = blockDim.x / 32;
  const int tiles = p.chunk / 16;
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x) {
    size_t block = linear_block;
    const int group = block % (size_t)p.G;
    block /= (size_t)p.G;
    const int chunk_idx = block % (size_t)p.nchunks;
    const int b = block / (size_t)p.nchunks;
    const int t0 = chunk_idx * p.chunk;
    for (int tile = warp; tile < tiles * tiles; tile += warps) {
      const int row = (tile / tiles) * 16;
      const int col = (tile % tiles) * 16;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
      for (int n0 = 0; n0 < p.N; n0 += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major>
            c_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::col_major>
            b_frag;
        const __nv_bfloat16 *c_ptr =
            reinterpret_cast<const __nv_bfloat16 *>(
                p.c + ssm2_bc_idx(b, t0 + row, group, n0, p.T, p.G, p.N));
        const __nv_bfloat16 *b_ptr =
            reinterpret_cast<const __nv_bfloat16 *>(
                p.b + ssm2_bc_idx(b, t0 + col, group, n0, p.T, p.G, p.N));
        wmma::load_matrix_sync(c_frag, c_ptr, p.G * p.N);
        wmma::load_matrix_sync(b_frag, b_ptr, p.G * p.N);
        wmma::mma_sync(acc, c_frag, b_frag, acc);
      }
      wmma::store_matrix_sync(
          p.cb + cb_idx(p, b, chunk_idx, group, row, col), acc, p.chunk,
          wmma::mem_row_major);
    }
  }
}

// One warp computes one [16 tokens, 16 head channels] output tile. The first
// MMA is BF16 C @ H^T. The causal interaction term is TF32 as in cuTile.
__device__ __forceinline__ void out_wmma_tile(const SSM2Params &p,
                                              size_t linear_block, float *sm) {
  size_t block = linear_block;
  const int row_tile = block % (size_t)(p.chunk / 16);
  block /= (size_t)(p.chunk / 16);
  const int ptile = block % (size_t)(p.P / 16);
  block /= (size_t)(p.P / 16);
  const int chunk_idx = block % (size_t)p.nchunks;
  const size_t bh = block / (size_t)p.nchunks;
  const int b = bh / (size_t)p.H;
  const int h = bh - (size_t)b * p.H;
  const int group = h / p.heads_per_group;
  const int t0 = chunk_idx * p.chunk;
  const int row0 = row_tile * 16;
  const int p0 = ptile * 16;
  const int lane = threadIdx.x;

  float *dtx = sm;                         // [chunk, 16], row-major
  float *acc_tile = dtx + p.chunk * 16;   // [16, 16]
  float *a_tile = acc_tile + 256;         // [16, 8], row-major

  for (int i = lane; i < p.chunk * 16; i += 32) {
    const int t = i / 16;
    const int pi = i - t * 16;
    const float dtv =
        ssm2_from_bf16(p.dt[ssm2_dt_idx(b, t0 + t, h, p.T, p.H)]);
    const float xv = ssm2_from_bf16(
        p.x[ssm2_x_idx(b, t0 + t, h, p0 + pi, p.T, p.H, p.P)]);
    dtx[i] = ssm2_round_bf16(dtv * xv);
  }
  __syncwarp();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.f);
  for (int n0 = 0; n0 < p.N; n0 += 16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                   wmma::row_major>
        c_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                   wmma::col_major>
        h_frag;
    const __nv_bfloat16 *c_ptr = reinterpret_cast<const __nv_bfloat16 *>(
        p.c + ssm2_bc_idx(b, t0 + row0, group, n0, p.T, p.G, p.N));
    const __nv_bfloat16 *h_ptr = reinterpret_cast<const __nv_bfloat16 *>(
        p.hiddens +
        ssm2_hid_idx(b, h, chunk_idx, p0, n0, p.H, p.nchunks, p.P, p.N));
    wmma::load_matrix_sync(c_frag, c_ptr, p.G * p.N);
    wmma::load_matrix_sync(h_frag, h_ptr, p.N);
    wmma::mma_sync(acc, c_frag, h_frag, acc);
  }

  wmma::store_matrix_sync(acc_tile, acc, 16, wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < 256; i += 32) {
    const int row = i / 16;
    const float dec =
        p.decay[ssm2_dt_idx(b, t0 + row0 + row, h, p.T, p.H)];
    acc_tile[i] *= exp2f(dec);
  }
  __syncwarp();
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_tf32;
  wmma::load_matrix_sync(acc_tf32, acc_tile, 16, wmma::mem_row_major);

  for (int k0 = 0; k0 < p.chunk; k0 += 8) {
    for (int i = lane; i < 16 * 8; i += 32) {
      const int row = i / 8;
      const int k = i - row * 8;
      const int j = k0 + k;
      float value = 0.f;
      if (j <= row0 + row) {
        const float di =
            p.decay[ssm2_dt_idx(b, t0 + row0 + row, h, p.T, p.H)];
        const float dj = p.decay[ssm2_dt_idx(b, t0 + j, h, p.T, p.H)];
        value = ssm2_round_tf32(
            p.cb[cb_idx(p, b, chunk_idx, group, row0 + row, j)] *
            exp2f(di - dj));
      }
      a_tile[i] = value;
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                   wmma::row_major>
        b_frag;
    wmma::load_matrix_sync(a_frag, a_tile, 8);
    wmma::load_matrix_sync(b_frag, dtx + k0 * 16, 16);
    wmma::mma_sync(acc_tf32, a_frag, b_frag, acc_tf32);
    __syncwarp();
  }

  wmma::store_matrix_sync(acc_tile, acc_tf32, 16, wmma::mem_row_major);
  __syncwarp();
  const float skip = p.dskip[h];
  for (int i = lane; i < 256; i += 32) {
    const int row = i / 16;
    const int pi = i - row * 16;
    const size_t idx =
        ssm2_x_idx(b, t0 + row0 + row, h, p0 + pi, p.T, p.H, p.P);
    const float xv = ssm2_from_bf16(p.x[idx]);
    p.out[idx] = ssm2_to_bf16(acc_tile[i] + skip * xv);
  }
  __syncwarp();
}

__global__ void fwd_wmma_kernel(SSM2Params p, size_t blocks) {
  extern __shared__ unsigned char storage[];
  ssm2_bf16 *weighted_b = reinterpret_cast<ssm2_bf16 *>(storage);
  float *state_tile = reinterpret_cast<float *>(
      weighted_b + (size_t)p.chunk * 16);
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x)
    fwd_wmma_tile(p, linear_block, weighted_b, state_tile);
}

__global__ void cb_wmma_kernel(SSM2Params p, size_t blocks) {
  cb_wmma_tiles(p, blocks);
}

__global__ void out_wmma_kernel(SSM2Params p, size_t blocks) {
  extern __shared__ float sm[];
  for (size_t linear_block = blockIdx.x; linear_block < blocks;
       linear_block += gridDim.x)
    out_wmma_tile(p, linear_block, sm);
}

#endif
