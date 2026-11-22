// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#define FINAL_MASK 0xffffffff
#define MAX_THREADS_PER_BLOCK 1024
#define THREADS_PER_WARP 32

// Sum
template <typename T> struct SumOp {
    __device__ static T init() { return T(0); }
    __device__ static T op(const T &x, const T &y) { return x + y; }
};


/**
 * Warp Reduce and Block Reduce
 */
template <template <class> class Func, typename T>
__device__  T WarpReduce(T val) {
#pragma unroll
    for (int offset = THREADS_PER_WARP >> 1; offset > 0; offset >>= 1) {
        T tmp = __shfl_xor_sync(FINAL_MASK, val, offset);
        val   = Func<T>::op(tmp, val);
    }
    return val;
}

template <template <class> class Func, typename T>
__device__ T BlockReduce(const T &val) {
    constexpr int MAX_NUM_WARPS = MAX_THREADS_PER_BLOCK / THREADS_PER_WARP;
    const int     num_warps     = (blockDim.x + THREADS_PER_WARP - 1) / THREADS_PER_WARP;

    __shared__ T smem[MAX_NUM_WARPS];
    const int    warp_id = threadIdx.x / THREADS_PER_WARP;
    const int    lane_id = threadIdx.x % THREADS_PER_WARP;

    T val_reg = Func<T>::init();
    val_reg   = Func<T>::op(val_reg, val);
    val_reg   = WarpReduce<Func, T>(val_reg);
    if (lane_id == 0) {
        smem[warp_id] = val_reg;
    }
    __syncthreads();
    if (warp_id == 0) {
        val_reg = (lane_id < num_warps) ? smem[lane_id] : Func<T>::init();
        val_reg = WarpReduce<Func, T>(val_reg);
        if (lane_id == 0)
            smem[0] = val_reg;
    }
    __syncthreads();
    T result = smem[0];
    __syncthreads();
    return result;
}
