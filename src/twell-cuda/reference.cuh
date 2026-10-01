// Reference implementation of the gated FFN, used by main.cu to
// validate the TwELL Hopper kernels
#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define TG_M 128
#define TG_N 128
#define TG_K 8
#define TG_TM 8
#define TG_TN 8
#define TG_THREADS ((TG_M / TG_TM) * (TG_N / TG_TN))  // 256

// vectorized 128-bit load of 8 consecutive bf16 -> 8 floats
__device__ __forceinline__ void load8_bf16(const __nv_bfloat16* p, float out[8]) {
    int4 raw = *reinterpret_cast<const int4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(h[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

// C = A * B^T ; A (M x K) bf16 row-major, B (N x K) bf16 row-major, C (M x N) fp32.
// Used for gate = A@GATE^T and up = A@UP^T (both operands K-contiguous).
__global__ __launch_bounds__(TG_THREADS) void gemm_nt_bf16(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
    const int bm = blockIdx.y * TG_M;
    const int bn = blockIdx.x * TG_N;

    __shared__ float As[TG_K][TG_M];  // transposed: As[k][m]
    __shared__ float Bs[TG_K][TG_N];  // transposed: Bs[k][n]

    const int tid = threadIdx.x;
    const int tRow = tid / (TG_N / TG_TN);  // 0..15
    const int tCol = tid % (TG_N / TG_TN);  // 0..15

    // Threads 0..127 (rowSel 0) load 8 k of one A row; 128..255 load one B row.
    const int row = tid % TG_M;     // 0..127
    const int rowSel = tid / TG_M;  // 0 (A) or 1 (B)
    const __nv_bfloat16* srcBase =
        (rowSel == 0) ? &A[static_cast<long>(bm + row) * K]
                      : &B[static_cast<long>(bn + row) * K];

    float acc[TG_TM][TG_TN];
#pragma unroll
    for (int i = 0; i < TG_TM; ++i)
#pragma unroll
        for (int j = 0; j < TG_TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TG_K) {
        float pf[8];
        load8_bf16(srcBase + k0, pf);
        if (rowSel == 0) {
#pragma unroll
            for (int t = 0; t < 8; ++t) As[t][row] = pf[t];
        } else {
#pragma unroll
            for (int t = 0; t < 8; ++t) Bs[t][row] = pf[t];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TG_K; ++kk) {
            float ra[TG_TM], rb[TG_TN];
#pragma unroll
            for (int i = 0; i < TG_TM; i += 4) {
                float4 v = *reinterpret_cast<const float4*>(&As[kk][tRow * TG_TM + i]);
                ra[i] = v.x; ra[i + 1] = v.y; ra[i + 2] = v.z; ra[i + 3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < TG_TN; j += 4) {
                float4 v = *reinterpret_cast<const float4*>(&Bs[kk][tCol * TG_TN + j]);
                rb[j] = v.x; rb[j + 1] = v.y; rb[j + 2] = v.z; rb[j + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TG_TM; ++i)
#pragma unroll
                for (int j = 0; j < TG_TN; ++j)
                    acc[i][j] = __fmaf_rn(ra[i], rb[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TG_TM; ++i) {
        const int m = bm + tRow * TG_TM + i;
#pragma unroll
        for (int j = 0; j < TG_TN; ++j)
            C[static_cast<long>(m) * N + bn + tCol * TG_TN + j] = acc[i][j];
    }
}

// OUT = H * DOWN ; H (M x K) fp32 row-major, DOWN (K x hidden) bf16 row-major,
// OUT (M x hidden) bf16. K = intermediate. Used for the down projection.
__global__ __launch_bounds__(TG_THREADS) void gemm_nn_down(
    const float* __restrict__ H, const __nv_bfloat16* __restrict__ DOWN,
    __nv_bfloat16* __restrict__ OUT, int M, int hidden, int K) {
    const int bm = blockIdx.y * TG_M;  // rows m
    const int bn = blockIdx.x * TG_N;  // cols o (into hidden)

    __shared__ float As[TG_K][TG_M];  // H tile:   As[k][m]
    __shared__ float Bs[TG_K][TG_N];  // DOWN tile: Bs[k][o]

    const int tid = threadIdx.x;
    const int tRow = tid / (TG_N / TG_TN);
    const int tCol = tid % (TG_N / TG_TN);

    // H load: 4 contiguous k (float4) per thread, k-contiguous in H.
    const int hRow = tid / (TG_K / 4);      // TG_K/4 = 2 -> hRow 0..127
    const int hK = (tid % (TG_K / 4)) * 4;  // 0 or 4
    // DOWN load: contiguous o along tid; TG_N=128 cols, 256 threads -> 2 k rows.
    const int dCol = tid % TG_N;            // 0..127
    const int dRow = tid / TG_N;            // 0 or 1
    const int dStride = TG_THREADS / TG_N;  // 2

    float acc[TG_TM][TG_TN];
#pragma unroll
    for (int i = 0; i < TG_TM; ++i)
#pragma unroll
        for (int j = 0; j < TG_TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TG_K) {
        float4 hv = *reinterpret_cast<const float4*>(
            &H[static_cast<long>(bm + hRow) * K + k0 + hK]);
        As[hK + 0][hRow] = hv.x;
        As[hK + 1][hRow] = hv.y;
        As[hK + 2][hRow] = hv.z;
        As[hK + 3][hRow] = hv.w;
#pragma unroll
        for (int off = 0; off < TG_K; off += dStride)
            Bs[dRow + off][dCol] = __bfloat162float(
                DOWN[static_cast<long>(k0 + dRow + off) * hidden + bn + dCol]);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TG_K; ++kk) {
            float ra[TG_TM], rb[TG_TN];
#pragma unroll
            for (int i = 0; i < TG_TM; i += 4) {
                float4 v = *reinterpret_cast<const float4*>(&As[kk][tRow * TG_TM + i]);
                ra[i] = v.x; ra[i + 1] = v.y; ra[i + 2] = v.z; ra[i + 3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < TG_TN; j += 4) {
                float4 v = *reinterpret_cast<const float4*>(&Bs[kk][tCol * TG_TN + j]);
                rb[j] = v.x; rb[j + 1] = v.y; rb[j + 2] = v.z; rb[j + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TG_TM; ++i)
#pragma unroll
                for (int j = 0; j < TG_TN; ++j)
                    acc[i][j] = __fmaf_rn(ra[i], rb[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TG_TM; ++i) {
        const int m = bm + tRow * TG_TM + i;
#pragma unroll
        for (int j = 0; j < TG_TN; ++j)
            OUT[static_cast<long>(m) * hidden + bn + tCol * TG_TN + j] =
                __float2bfloat16(acc[i][j]);
    }
}

// H[i] = bf16(relu(gate[i])) * up[i] ; active[i] = gate[i] > 0
__global__ void fuse_gate_up_kernel(const float* __restrict__ gate_raw,
                                    const float* __restrict__ up_raw,
                                    float* __restrict__ H,
                                    uint8_t* __restrict__ active, long total) {
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float gp = gate_raw[idx];
    bool pos = gp > 0.0f;
    float gate = pos ? __bfloat162float(__float2bfloat16(gp)) : 0.0f;
    H[idx] = gate * up_raw[idx];
    active[idx] = pos ? 1u : 0u;
}

// launches the portable reference gated FFN: gate + up GEMMs, fuse, down GEMM.
static void launch_reference(const __nv_bfloat16* A, const __nv_bfloat16* GATE,
                             const __nv_bfloat16* UP, const __nv_bfloat16* DOWN,
                             float* gate_raw, float* up_raw, float* H,
                             uint8_t* active, __nv_bfloat16* OUT, int M,
                             int hidden, int intermediate) {
    dim3 block(TG_THREADS);
    dim3 grid_nt(intermediate / TG_N, M / TG_M);
    gemm_nt_bf16<<<grid_nt, block>>>(A, GATE, gate_raw, M, intermediate, hidden);
    gemm_nt_bf16<<<grid_nt, block>>>(A, UP, up_raw, M, intermediate, hidden);
    long total = static_cast<long>(M) * intermediate;
    fuse_gate_up_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(
        gate_raw, up_raw, H, active, total);
    dim3 grid_nn(hidden / TG_N, M / TG_M);
    gemm_nn_down<<<grid_nn, block>>>(H, DOWN, OUT, M, hidden, intermediate);
}

// per (row, 256-tile) count of positive gate activations
__global__ void tile_count_kernel(const uint8_t* __restrict__ active,
                                  int* __restrict__ counts, int M,
                                  int intermediate) {
    const int tiles = intermediate / 256;
    long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long total = static_cast<long>(M) * tiles;
    if (t >= total) return;
    int m = static_cast<int>(t / tiles);
    int ti = static_cast<int>(t % tiles);
    const uint8_t* row = active + static_cast<long>(m) * intermediate + ti * 256;
    int c = 0;
    for (int i = 0; i < 256; ++i) c += row[i];
    counts[t] = c;
}
