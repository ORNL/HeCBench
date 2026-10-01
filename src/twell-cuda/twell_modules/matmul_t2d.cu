#include "torch_compat.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#define WARP_SIZE 32
#define STRIDE_8xWARP 256

namespace TWELL_T2D {

// computes one split of one dense output row from one packed input row.
// inputs: packed input row, downprojection weights, output row buffer. output: updated OUT_d row.
template <
    // full tile size
    const int T_n,
    // compressed tile size
    const int T_n_compressed,
    // number of tiles
    const int NUM_T_n,
    const int OUT_DIM,
    const int SPLIT_OUT_DIM
>
__global__ __launch_bounds__(32) void mm_t2d_kernel(
    // IN_DIM x COMPRESSED_FEATURE_DIM
    const uint32_t* IN_twell_packed_d,
    // FEATURE_DIM x OUT_DIM
    const __nv_bfloat16* DOWN_d,
    // IN_DIM x OUT_DIM
    __nv_bfloat16* OUT_d
)
{
    static_assert((SPLIT_OUT_DIM % STRIDE_8xWARP) == 0, "OUT_DIM must be divisible by WARP_SIZE.");
    static_assert((OUT_DIM % SPLIT_OUT_DIM) == 0, "OUT_DIM must be divisible by SPLIT_OUT_DIM.");
    static_assert(T_n_compressed == WARP_SIZE,
        "Warp sync twell 2 dense optimized for when the compressed tile is 32-dim.");
    float OUT_accum[OUT_DIM / STRIDE_8xWARP][8] = {0.0f};

    // each block computes one row
    constexpr int NUM_LOAD_ITERS = SPLIT_OUT_DIM / STRIDE_8xWARP;

    // shift by thread idx: each lane in the 32-thread block gets one element of the compressed tile
    IN_twell_packed_d += static_cast<size_t>(blockIdx.x) * T_n_compressed * NUM_T_n + threadIdx.x;
    // each iteration, each thread loads/stores 8 contiguous features
    // such that the whole warp processes STRIDE_8xWARP features
    DOWN_d += threadIdx.x * 8 + blockIdx.y * SPLIT_OUT_DIM;
    OUT_d += static_cast<size_t>(blockIdx.x) * OUT_DIM + threadIdx.x * 8 + blockIdx.y * SPLIT_OUT_DIM;
    #pragma unroll 1
    for(int tile_idx = 0; tile_idx < NUM_T_n; ++tile_idx)
    {
        // each thread gets one scalar from the packed tile, giving the warp one coalesced access
        const int lane_tile_register = __ldcs(&IN_twell_packed_d[tile_idx * T_n_compressed]);
        const int num_nonzeros = __shfl_sync(0xFFFFFFFFu, lane_tile_register, 0);
        #pragma unroll 1
        for (int idx = 1; idx < num_nonzeros + 1; ++idx)
        {
            const uint32_t compressed_idx_bf16 = __shfl_sync(0xFFFFFFFFu, lane_tile_register, idx);
            const uint32_t nonzero_idx = compressed_idx_bf16 & 0xFFFFu; // lower 16 bits for index
            const __nv_bfloat162 nonzero_feature = __bfloat162bfloat162(
                reinterpret_cast<const __nv_bfloat16*>(
                    &compressed_idx_bf16)[1]
            ); // upper 16 bits for feature
            #pragma unroll 
            // load 8 elements at a time
            for(int iter_idx = 0; iter_idx < NUM_LOAD_ITERS; iter_idx += 1)
            {
                const uint4 packed_bfloats_x8 = *reinterpret_cast<const uint4*>(
                    DOWN_d + nonzero_idx * OUT_DIM + iter_idx * STRIDE_8xWARP);
                // unpack the 8 bfloat16s
                const __nv_bfloat162 packed_bfloats_1 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.x); 
                __nv_bfloat162 scaled_bfloats_1 = __hmul2(nonzero_feature, packed_bfloats_1);
                float2 scaled_floats_1 = __bfloat1622float2(scaled_bfloats_1);
                // accumulate the dot product of the nonzero feature with these 8 features
                OUT_accum[iter_idx][0] += scaled_floats_1.x;
                OUT_accum[iter_idx][1] += scaled_floats_1.y;
                const __nv_bfloat162 packed_bfloats_2 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.y); 
                __nv_bfloat162 scaled_bfloats_2 = __hmul2(nonzero_feature, packed_bfloats_2);
                float2 scaled_floats_2 = __bfloat1622float2(scaled_bfloats_2);
                OUT_accum[iter_idx][2] += scaled_floats_2.x;
                OUT_accum[iter_idx][3] += scaled_floats_2.y;

                const __nv_bfloat162 packed_bfloats_3 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.z);
                __nv_bfloat162 scaled_bfloats_3 = __hmul2(nonzero_feature, packed_bfloats_3);
                float2 scaled_floats_3 = __bfloat1622float2(scaled_bfloats_3);
                OUT_accum[iter_idx][4] += scaled_floats_3.x;
                OUT_accum[iter_idx][5] += scaled_floats_3.y;

                const __nv_bfloat162 packed_bfloats_4 = *reinterpret_cast<const __nv_bfloat162*>(&packed_bfloats_x8.w);
                __nv_bfloat162 scaled_bfloats_4 = __hmul2(nonzero_feature, packed_bfloats_4);
                float2 scaled_floats_4 = __bfloat1622float2(scaled_bfloats_4);
                OUT_accum[iter_idx][6] += scaled_floats_4.x;
                OUT_accum[iter_idx][7] += scaled_floats_4.y;
            }
        }
    }
    #pragma unroll
    for(int iter_idx = 0; iter_idx < NUM_LOAD_ITERS; iter_idx += 1)
    {
        __nv_bfloat162  __align__(8) packed_bfloats_x8[4];
        packed_bfloats_x8[0] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][0], OUT_accum[iter_idx][1]);
        packed_bfloats_x8[1] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][2], OUT_accum[iter_idx][3]);
        packed_bfloats_x8[2] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][4], OUT_accum[iter_idx][5]);
        packed_bfloats_x8[3] = __floats2bfloat162_rn(
            OUT_accum[iter_idx][6], OUT_accum[iter_idx][7]);
        
        __stcs(reinterpret_cast<uint4*>(OUT_d + iter_idx * STRIDE_8xWARP),
               *reinterpret_cast<uint4*>(packed_bfloats_x8));
    }
}

// launches the packed-to-dense downprojection kernel.
// inputs: packed input, downprojection weights, output buffer, logical dims, split count.
// output: OUT filled on the current CUDA stream.
void mm_t2d_wid(
    uint32_t* in_packed,
    at::BFloat16* down_weight,
    at::BFloat16* out,
    const size_t IN_DIM,
    const int FEATURE_DIM,
    const int OUT_DIM,
    const int NUM_SPLITS = 2
) {
    if (IN_DIM == 0 || IN_DIM > 0x7fffffffu || FEATURE_DIM <= 0 || OUT_DIM <= 0) {
        return;
    }
    if ((FEATURE_DIM % 256) != 0) {
        printf("mm_t2d_wid: FEATURE_DIM must be divisible by 256 (got %d)\n", FEATURE_DIM);
        return;
    }
    if (OUT_DIM != 2048) {
        printf("mm_t2d_wid: only OUT_DIM=2048 is supported (got %d)\n", OUT_DIM);
        return;
    }

    constexpr int T_n = 256;
    constexpr int T_n_compressed = T_n / 8;
    const int NUM_T_n = FEATURE_DIM / T_n;

    // [&] captures outer context for the lambda; auto is the only way to pass
    // different compile-time constants
    auto num_tiles_lambda = [&] (auto TYPE_NUM_SPLITS, auto TYPE_NUM_T_n)
    {
        constexpr int CONSTEXPR_NUM_SPLITS = decltype(TYPE_NUM_SPLITS)::value;
        constexpr int CONSTEXPR_NUM_T_n = decltype(TYPE_NUM_T_n)::value;
        constexpr int SPLIT_OUT_DIM = 2048 / CONSTEXPR_NUM_SPLITS;

        // each warp computes one row
        dim3 block_dim(32, 1, 1);
        dim3 grid_dim(static_cast<unsigned int>(IN_DIM), CONSTEXPR_NUM_SPLITS, 1);
        mm_t2d_kernel<T_n, T_n_compressed, CONSTEXPR_NUM_T_n, 2048, SPLIT_OUT_DIM>
            <<<grid_dim, block_dim>>>(
                in_packed,
                reinterpret_cast<const __nv_bfloat16*>(down_weight),
                reinterpret_cast<__nv_bfloat16*>(out)            );
    };

    auto num_warps_lambda = [&] (auto TYPE_NUM_T_n)
    {
        switch(NUM_SPLITS)
        {
            case 1: num_tiles_lambda(std::integral_constant<int, 1>{}, TYPE_NUM_T_n); break;
            case 2: num_tiles_lambda(std::integral_constant<int, 2>{}, TYPE_NUM_T_n); break;
            case 4: num_tiles_lambda(std::integral_constant<int, 4>{}, TYPE_NUM_T_n); break;
            case 8: num_tiles_lambda(std::integral_constant<int, 8>{}, TYPE_NUM_T_n); break;
            default:
                printf("mm_t2d_wid: unsupported NUM_SPLITS=%d (supported: 1/2/4/8)\n", NUM_SPLITS);
                return;
        }
    };

    switch (NUM_T_n) {
        case 22:
            num_warps_lambda(std::integral_constant<int, 22>{});
            break;
        case 32:
            num_warps_lambda(std::integral_constant<int, 32>{});
            break;
        default:
            printf("mm_t2d_wid: unsupported NUM_T_n=%d from FEATURE_DIM=%d (supported: 22/32)\n", NUM_T_n, FEATURE_DIM);
            return;
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("mm_t2d_wid kernel launch failed: %s\n", cudaGetErrorString(status));
    }
}

}  // namespace TWELL_T2D

#undef WARP_SIZE
#undef STRIDE_8xWARP
