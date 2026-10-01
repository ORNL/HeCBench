#include <cassert>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <cuda_bf16.h>
#include "torch_compat.h"
#include "hilbert.cu"

#define WARP_SIZE 32
#define WARP_GROUP_SIZE 128
// 2 consumer and 1 producer warpgroups
#define NUM_THREADS_PER_BLOCK (WARP_GROUP_SIZE*3)
#define WGMMA_m 64
#define WGMMA_k 16
#define NUM_SMs 132

namespace TWELL_D2T {

// encodes a shared-memory address or stride fragment for a WGMMA descriptor.
// input: raw shared-memory address/stride bits. output: descriptor field bits.
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x)
{
    return (x & 0x3FFFF) >> 0x4;
}

// builds a shared-memory descriptor for a swizzled bf16 tile.
// input: base pointer to shared-memory tile. output: packed descriptor for wgmma.
__device__ uint64_t make_smem_descriptor(__nv_bfloat16* ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(16)) << 16;
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(1024)) << 32;
    desc |= 1llu << 62;
    return desc;
}

// issues one m64n256k16 bf16xbf16->f32 WGMMA step.
// inputs: accumulator fragment C, shared-memory tiles A_s/B_s. output: updated C.
// https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma
template<int C_scale, int A_scale, int B_scale, int A_trans, int B_trans>
__device__ __forceinline__ void wgmma256(float C[16][8], __nv_bfloat16* A_s, __nv_bfloat16* B_s) {
    uint64_t A_desc = make_smem_descriptor(&A_s[0]);
    uint64_t B_desc = make_smem_descriptor(&B_s[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130,    %131,  %132,  %133,  %134;\n"
        "}\n"
        :   "+f"(C[0][0]), "+f"(C[0][1]), "+f"(C[0][2]), "+f"(C[0][3]), "+f"(C[0][4]), "+f"(C[0][5]), "+f"(C[0][6]), "+f"(C[0][7]),
            "+f"(C[1][0]), "+f"(C[1][1]), "+f"(C[1][2]), "+f"(C[1][3]), "+f"(C[1][4]), "+f"(C[1][5]), "+f"(C[1][6]), "+f"(C[1][7]),
            "+f"(C[2][0]), "+f"(C[2][1]), "+f"(C[2][2]), "+f"(C[2][3]), "+f"(C[2][4]), "+f"(C[2][5]), "+f"(C[2][6]), "+f"(C[2][7]),
            "+f"(C[3][0]), "+f"(C[3][1]), "+f"(C[3][2]), "+f"(C[3][3]), "+f"(C[3][4]), "+f"(C[3][5]), "+f"(C[3][6]), "+f"(C[3][7]),
            "+f"(C[4][0]), "+f"(C[4][1]), "+f"(C[4][2]), "+f"(C[4][3]), "+f"(C[4][4]), "+f"(C[4][5]), "+f"(C[4][6]), "+f"(C[4][7]),
            "+f"(C[5][0]), "+f"(C[5][1]), "+f"(C[5][2]), "+f"(C[5][3]), "+f"(C[5][4]), "+f"(C[5][5]), "+f"(C[5][6]), "+f"(C[5][7]),
            "+f"(C[6][0]), "+f"(C[6][1]), "+f"(C[6][2]), "+f"(C[6][3]), "+f"(C[6][4]), "+f"(C[6][5]), "+f"(C[6][6]), "+f"(C[6][7]),
            "+f"(C[7][0]), "+f"(C[7][1]), "+f"(C[7][2]), "+f"(C[7][3]), "+f"(C[7][4]), "+f"(C[7][5]), "+f"(C[7][6]), "+f"(C[7][7]),
            "+f"(C[8][0]), "+f"(C[8][1]), "+f"(C[8][2]), "+f"(C[8][3]), "+f"(C[8][4]), "+f"(C[8][5]), "+f"(C[8][6]), "+f"(C[8][7]),
            "+f"(C[9][0]), "+f"(C[9][1]), "+f"(C[9][2]), "+f"(C[9][3]), "+f"(C[9][4]), "+f"(C[9][5]), "+f"(C[9][6]), "+f"(C[9][7]),
            "+f"(C[10][0]), "+f"(C[10][1]), "+f"(C[10][2]), "+f"(C[10][3]), "+f"(C[10][4]), "+f"(C[10][5]), "+f"(C[10][6]), "+f"(C[10][7]),
            "+f"(C[11][0]), "+f"(C[11][1]), "+f"(C[11][2]), "+f"(C[11][3]), "+f"(C[11][4]), "+f"(C[11][5]), "+f"(C[11][6]), "+f"(C[11][7]),
            "+f"(C[12][0]), "+f"(C[12][1]), "+f"(C[12][2]), "+f"(C[12][3]), "+f"(C[12][4]), "+f"(C[12][5]), "+f"(C[12][6]), "+f"(C[12][7]),
            "+f"(C[13][0]), "+f"(C[13][1]), "+f"(C[13][2]), "+f"(C[13][3]), "+f"(C[13][4]), "+f"(C[13][5]), "+f"(C[13][6]), "+f"(C[13][7]),
            "+f"(C[14][0]), "+f"(C[14][1]), "+f"(C[14][2]), "+f"(C[14][3]), "+f"(C[14][4]), "+f"(C[14][5]), "+f"(C[14][6]), "+f"(C[14][7]),
            "+f"(C[15][0]), "+f"(C[15][1]), "+f"(C[15][2]), "+f"(C[15][3]), "+f"(C[15][4]), "+f"(C[15][5]), "+f"(C[15][6]), "+f"(C[15][7])
        : "l"(A_desc), "l"(B_desc), "n"(int32_t(C_scale)), "n"(int32_t(A_scale)),
            "n"(int32_t(B_scale)), "n"(int32_t(A_trans)), "n"(int32_t(B_trans)));
}

// dispatches the supported WGMMA shape.
// inputs: accumulator fragment C, shared-memory tiles A_s/B_s. output: updated C.
template<int WGMMA_n, int C_scale, int A_scale, int B_scale,
         int A_trans, int B_trans>
__device__ __forceinline__ void wgmma(
    float c[WGMMA_n / 16][8], __nv_bfloat16* A_s, __nv_bfloat16* B_s)
{
    static_assert(WGMMA_n == 256);
    if constexpr (WGMMA_n == 256) {
        wgmma256<C_scale, A_scale, B_scale, A_trans, B_trans>(c, A_s, B_s);
    }
}

// initializes a block-scoped mbarrier in shared memory.
// inputs: barrier storage pointer, expected arrival count. output: initialized barrier.
__device__ __forceinline__ void ptx_init_smem_barrier(
    uint64_t* barrier_pointer, int barrier_count)
{
    int32_t barrier_pointer_smem = static_cast<int32_t>(
        __cvta_generic_to_shared(barrier_pointer));
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(barrier_pointer_smem), "r"(barrier_count));
}

// arrives on an mbarrier and declares the number of bytes a TMA transfer will fill.
// inputs: barrier storage pointer, bytes to transfer. output: barrier state updated.
__device__ __forceinline__ void ptx_arrive_tx_smem_barrier(
    uint64_t* barrier_pointer, uint32_t bytes_to_transfer)
{
    int32_t barrier_pointer_smem = static_cast<int32_t>(
        __cvta_generic_to_shared(barrier_pointer));
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(barrier_pointer_smem), "r"(bytes_to_transfer));
}

// launches a 2D TMA load from global memory into shared memory.
// inputs: destination smem tile, source tensor map, source tile offsets, completion barrier.
// output: async copy scheduled and completion credited to barrier.
__device__ __forceinline__ void ptx_load_tile_tma_2d(
    __nv_bfloat16* dst_pointer,
    const void* const src_tensor_map_pointer,
    uint32_t src_offset_dim0,
    uint32_t src_offset_dim1,
    uint64_t* barrier_pointer)
{
    int32_t dst_pointer_smem = static_cast<int32_t>(
        __cvta_generic_to_shared(dst_pointer));
    int64_t src_tensor_map_pointer_gmem = reinterpret_cast<int64_t>(
        src_tensor_map_pointer);
    int32_t barrier_pointer_smem = static_cast<int32_t>(
        __cvta_generic_to_shared(barrier_pointer));

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(dst_pointer_smem), "l"(src_tensor_map_pointer_gmem),
          "r"(src_offset_dim1), "r"(src_offset_dim0),
          "r"(barrier_pointer_smem)
        : "memory");
}

// launches a multicast 2D TMA load into the blocks selected by cluster_mask.
// inputs: destination smem tile, source tensor map, source tile offsets, cluster mask,
// completion barrier. output: async multicast copy scheduled.
__device__ __forceinline__ void ptx_load_tile_tma_multicast_2d(
    __nv_bfloat16* dst_pointer,
    const void* const src_tensor_map_pointer,
    uint32_t src_offset_dim0,
    uint32_t src_offset_dim1,
    uint16_t cluster_mask,
    uint64_t* barrier_pointer)
{
    int32_t dst_pointer_smem = static_cast<int32_t>(
        __cvta_generic_to_shared(dst_pointer));
    int64_t src_tensor_map_pointer_gmem = reinterpret_cast<int64_t>(
        src_tensor_map_pointer);
    int32_t barrier_pointer_smem = static_cast<int32_t>(
        __cvta_generic_to_shared(barrier_pointer));

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(dst_pointer_smem), "l"(src_tensor_map_pointer_gmem),
          "r"(src_offset_dim1), "r"(src_offset_dim0),
          "r"(barrier_pointer_smem), "h"(cluster_mask)
        : "memory");
}

// launches a transposed 3D TMA store from shared memory into global memory.
// inputs: destination tensor map, source smem tile, destination tile offsets.
// output: async store scheduled in the current bulk group.
template <typename T = __nv_bfloat16, int last_dim_size = 64>
__device__ __forceinline__ void ptx_store_transposed_tile_tma_3d(
    const void* const dst_tensor_map_pointer,
    T* src_pointer,
    uint32_t dst_offset_dim0,
    uint32_t dst_offset_dim1)
{
    int64_t dst_tensor_map_pointer_gmem = reinterpret_cast<int64_t>(
        dst_tensor_map_pointer);
    int32_t src_pointer_smem = static_cast<int32_t>(
        __cvta_generic_to_shared(src_pointer));

    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group "
        " [%0, {0, %1, %2}], [%3];"
        :
        : "l"(dst_tensor_map_pointer_gmem),
          "r"(dst_offset_dim0), "r"(dst_offset_dim1 / last_dim_size),
          "r"(src_pointer_smem)
        : "memory");
}

// spins until an mbarrier reaches the requested parity.
// inputs: barrier storage pointer, phase bit. output: none.
__device__ __forceinline__ void ptx_wait_barrier(
    uint64_t* barrier_pointer, uint32_t phase_bit)
{
    int32_t barrier_pointer_smem = static_cast<int32_t>(
        __cvta_generic_to_shared(barrier_pointer));

    asm volatile(
      "{\n"
        ".reg .pred P1;\n"
      "WAIT_START:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
        "P1, [%0], %1;\n"
        "@P1 bra.uni WAIT_END;\n"
        "bra.uni WAIT_START;\n"
      "WAIT_END:\n"
      "}\n"
        :: "r"(barrier_pointer_smem), "r"(phase_bit));
}

// arrives on a shared-memory mbarrier located in another block of the cluster.
// inputs: barrier storage pointer, remote block rank in the cluster, arrival count.
// output: remote barrier state updated.
__device__ __forceinline__ void ptx_arrive_barrier_across_cluster(
    uint64_t* barrier_pointer, uint32_t remote_cluster_lane, uint32_t count)
{
    int32_t barrier_pointer_smem = static_cast<int32_t>(
        __cvta_generic_to_shared(barrier_pointer));

    asm volatile(
        "{\n"
        ".reg .b32 remAddr32;\n"
        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n"
        "mbarrier.arrive.shared::cluster.b64 _, [remAddr32], %2;\n"
        "}"
        :
        : "r"(barrier_pointer_smem), "r"(remote_cluster_lane), "r"(count)
        : "memory");
}

// shared-memory queue entry holding one A tile and one B tile.
// input/output: storage only.
template <const int T_m, const int T_n, const int T_k>
struct Tiles
{
    alignas(128) __nv_bfloat16 a[T_m][T_k];
    alignas(128) __nv_bfloat16 b[T_n][T_k];
};

// shared-memory state for the kernel.
// input/output: queue buffers plus packed C staging, where c_packed[row][0]
// stores the row nnz and the remaining entries store packed (idx, value).
template <const int T_m, const int T_n, const int T_k, const int QUEUE_SIZE,
          const int T_n_compressed, int PADDING = 4>
struct SmemStorage
{
    Tiles<T_m, T_n, T_k> queue[QUEUE_SIZE];
    alignas(128) uint32_t c_packed[T_m][T_n_compressed + PADDING];
};

// computes one scheduled 128x256 tile per consumer block using a producer/consumer queue.
// inputs: tensor maps for A/B/C, per-active-cluster schedule, schedule length, K.
// output: packed positive bf16 results written to C_packed_tm.
template <
    const int T_m,
    const int T_n,
    const int T_k,
    const int CLUSTER_DIM_m,
    const int CLUSTER_DIM_n,
    const int QUEUE_SIZE,
    const int NUM_ACTIVE_SMs,
    const int T_n_compressed,
    const bool LOOP_OVERFLOW_STORAGE
>
__global__ __launch_bounds__(NUM_THREADS_PER_BLOCK)
           __cluster_dims__(CLUSTER_DIM_m * CLUSTER_DIM_n, 1, 1)
void mm_wgmma_nt_kernel(
    const CUtensorMap __grid_constant__ A_tm,
    const CUtensorMap __grid_constant__ B_tm,
    const CUtensorMap __grid_constant__ C_packed_tm,
    const int* schedule_gmem_ptr,
    const int schedule_size_per_sm,
    const int K
)
{
    static_assert(
        (T_m == 64 * 2),
        "Only T_m == 128 supported"
    );
    constexpr int CLUSTER_SIZE = CLUSTER_DIM_m * CLUSTER_DIM_n;
    extern __shared__ __align__(1024) unsigned char dynamic_smem[];
    int cluster_idx;
    asm ("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_idx) :);
    int cluster_lane_m;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(cluster_lane_m) :);
    int cluster_lane_n = cluster_lane_m % CLUSTER_DIM_n;
    cluster_lane_m /= CLUSTER_DIM_n;

    SmemStorage<T_m, T_n, T_k, QUEUE_SIZE, T_n_compressed>& tiles_s = 
        *reinterpret_cast<SmemStorage<T_m, T_n, T_k, QUEUE_SIZE, T_n_compressed>*>(dynamic_smem);
    int* schedule_s = reinterpret_cast<int*>(
        dynamic_smem + sizeof(SmemStorage<T_m, T_n, T_k, QUEUE_SIZE, T_n_compressed>));

    schedule_gmem_ptr += cluster_idx * schedule_size_per_sm;
    if(threadIdx.x < schedule_size_per_sm) 
        schedule_s[threadIdx.x] = schedule_gmem_ptr[threadIdx.x];

    __syncthreads();

    __shared__ __align__(8) uint64_t queue_full[QUEUE_SIZE];
    __shared__ __align__(8) uint64_t queue_empty[QUEUE_SIZE];
    if(threadIdx.x == 0){
        #pragma unroll
        for (int queue_idx = 0; queue_idx < QUEUE_SIZE; queue_idx++)
        {
            // 1 producer thread, 2 consumer warp groups
            ptx_init_smem_barrier(&queue_full[queue_idx], 1);
            ptx_init_smem_barrier(&queue_empty[queue_idx], 2 * CLUSTER_SIZE);
        }
    }
    // waits for the whole cluster to ensure all barriers are initialized
    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);

    if (threadIdx.x < WARP_GROUP_SIZE)
    {
        // decrease to the minimum assignable registers in the producer path
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(24): "memory");
        if(threadIdx.x == 0)
        {
        
        int queue_idx = 0;
        int queue_phase = 0;
        uint16_t mask_multicast_m = 0;
        if constexpr (CLUSTER_DIM_m > 1)
        {
            for(int i = 0; i < CLUSTER_DIM_m; i++){
                // each value will occur every CLUSTER_DIM_n elements and be
                // offset by the nth lane, as there are CLUSTER_DIM_n possible
                // groups
                mask_multicast_m |= (1u << (i*CLUSTER_DIM_n));
            }
            mask_multicast_m <<= cluster_lane_n;
        }
        uint16_t mask_multicast_n;
        if constexpr (CLUSTER_DIM_n > 1)
        {
            // CLUSTER_DIM_n consecutive 1s offset by the group size
            mask_multicast_n = ((1u << CLUSTER_DIM_n) - 1) 
                            << (cluster_lane_m * CLUSTER_DIM_n);
        }

        for (int schedule_it = 0; schedule_it < schedule_size_per_sm; ++schedule_it)
        {
            const int packed_tile = schedule_s[schedule_it];
            if (packed_tile == -1) {
                break;
            }
            int tile_coord_m = packed_tile >> 16;
            int tile_coord_n = packed_tile & 0xFFFF;
            if constexpr (CLUSTER_DIM_n > 1)
            {
                tile_coord_n *= CLUSTER_DIM_n;
                tile_coord_n += cluster_lane_n;
            }
            if constexpr (CLUSTER_DIM_m > 1)
            {
                tile_coord_m *= CLUSTER_DIM_m;
                tile_coord_m += cluster_lane_m;
            }
            for (int tile_start_k = 0; tile_start_k < K; tile_start_k += T_k, queue_idx++)
            {
                // modulo loop
                if (queue_idx == QUEUE_SIZE)
                {
                    queue_idx = 0;
                    queue_phase ^= 1;
                }

                ptx_wait_barrier(&queue_empty[queue_idx], queue_phase);
                
                ptx_arrive_tx_smem_barrier(
                    &queue_full[queue_idx],
                    sizeof(tiles_s.queue[queue_idx].a) +
                    sizeof(tiles_s.queue[queue_idx].b)
                );
                
                if constexpr (CLUSTER_DIM_n > 1)
                {
                    if (cluster_lane_n == 0)
                    {
                        // only one warp group per cluster needs to load A
                        // different B tiles, working on different columns, can
                        // multicast the same A tile for the row
                        ptx_load_tile_tma_multicast_2d(
                            // destination tile from smem
                            &tiles_s.queue[queue_idx].a[0][0],
                            // tm pointer from gmem
                            &A_tm,
                            // start in dimensions
                            tile_coord_m*T_m,
                            tile_start_k,
                            // cluster mask for multicast
                            mask_multicast_n,
                            // barrier to update once data arrived
                            &queue_full[queue_idx]
                        );
                    }
                }
                else
                {
                    // if not multicasting, just load the tile like normal
                    ptx_load_tile_tma_2d(
                        // destination tile from smem
                        &tiles_s.queue[queue_idx].a[0][0],
                        // tm pointer from gmem
                        &A_tm,
                        // start in dimensions
                        tile_coord_m*T_m,
                        tile_start_k,
                        // barrier to update once data arrived
                        &queue_full[queue_idx]
                    );
                }
                
                if constexpr (CLUSTER_DIM_m > 1)
                {
                    if (cluster_lane_m == 0)
                    {
                        // only one warp group per cluster needs to load B
                        // different A tiles, working on different rows, can
                        // multicast the same B tile for the column
                        ptx_load_tile_tma_multicast_2d(
                            // destination tile from smem
                            &tiles_s.queue[queue_idx].b[0][0],
                            // tm pointer from gmem
                            &B_tm,
                            // start in dimensions
                            tile_coord_n*T_n,
                            tile_start_k,
                            // cluster mask for multicast
                            mask_multicast_m,
                            // barrier to update once data arrived
                            &queue_full[queue_idx]
                        );
                    }
                }
                else
                {
                    ptx_load_tile_tma_2d(
                        // destination tile from smem
                        &tiles_s.queue[queue_idx].b[0][0],
                        // tm pointer from gmem
                        &B_tm,
                        // start in dimensions
                        tile_coord_n*T_n,
                        tile_start_k,
                        // barrier to update once data arrived
                        &queue_full[queue_idx]
                    );
                }

            }
        }
        }
    }
    else
    {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(240): "memory");
        int queue_idx = 0;
        int queue_phase = 0;
        const int consumer_warpgroup_id = (threadIdx.x - WARP_GROUP_SIZE) / WARP_GROUP_SIZE;
        const int tile_start_m = consumer_warpgroup_id * WGMMA_m;
        const int consumer_thread_id = threadIdx.x % WARP_GROUP_SIZE;
        const uint thread_lane_idx_n = (consumer_thread_id % 32) % 4;

        // 4 warps divide the 64 rows of WGMMA_m, 16 each
        // within each 16 x T_n/16 slice, each thread processes
        // 4 * 2 elements in each 8 x 8 quadrant (row major partitioning)
        const int thread_store_offset_m = 
            // row offset
            (tile_start_m + consumer_thread_id / 32 * 16 +
             (consumer_thread_id % 32) / 4);
        
        const int thread_store_offset_n = 
            // column offset
            ((consumer_thread_id % 32) % 4) * 2;
        
        if (consumer_thread_id < CLUSTER_SIZE)
        {
            for (int queue_idx = 0; queue_idx < QUEUE_SIZE; queue_idx++)
            {
                // mark all slots as empty to the producer warp group in all
                // blocks in the cluster
                ptx_arrive_barrier_across_cluster(
                    &queue_empty[queue_idx],
                    consumer_thread_id,
                    1
                );
            }
        }

        float C_accum[T_n/16][8];
        for (int schedule_it = 0; schedule_it < schedule_size_per_sm; ++schedule_it)
        {
            const int packed_tile = schedule_s[schedule_it];
            if (packed_tile == -1) {
                break;
            }
            int tile_coord_m = packed_tile >> 16;
            int tile_coord_n = packed_tile & 0xFFFF;
            if constexpr (CLUSTER_DIM_n > 1)
            {
                tile_coord_n *= CLUSTER_DIM_n;
                tile_coord_n += cluster_lane_n;
            }
            if constexpr (CLUSTER_DIM_m > 1)
            {
                tile_coord_m *= CLUSTER_DIM_m;
                tile_coord_m += cluster_lane_m;
            }
            // first iteration with C_scale == 0 to reset accumulators
            if (queue_idx == QUEUE_SIZE)
            {
                queue_idx = 0;
                queue_phase ^= 1;
            }
            ptx_wait_barrier(&queue_full[queue_idx], queue_phase);
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            wgmma<T_n, 0, 1, 1, 0, 0>(
                C_accum,
                &tiles_s.queue[queue_idx].a[tile_start_m][0],
                &tiles_s.queue[queue_idx].b[0][0]
            );
            // rest of the iterations accumulating
            #pragma unroll
            for (int wgmma_start_k = WGMMA_k; wgmma_start_k < T_k; wgmma_start_k += WGMMA_k)
            {
                wgmma<T_n, 1, 1, 1, 0, 0>(
                    C_accum,
                    &tiles_s.queue[queue_idx].a[tile_start_m][wgmma_start_k],
                    &tiles_s.queue[queue_idx].b[0][wgmma_start_k]
                );
            }

            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(0): "memory");
            if (consumer_thread_id < CLUSTER_SIZE)
            {
                ptx_arrive_barrier_across_cluster(
                    &queue_empty[queue_idx],
                    consumer_thread_id,
                    1
                );
            }
            queue_idx++;
            // rest of the iterations with C_scale == 1
            for (int tile_idx_k = 1; tile_idx_k < K/T_k; tile_idx_k++, queue_idx++)
            {   
                if (queue_idx == QUEUE_SIZE)
                {
                    queue_idx = 0;
                    queue_phase ^= 1;
                }
                ptx_wait_barrier(&queue_full[queue_idx], queue_phase);

                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int wgmma_start_k = 0; wgmma_start_k < T_k; wgmma_start_k += WGMMA_k)
                {
                    wgmma<T_n, 1, 1, 1, 0, 0>(
                        C_accum,
                        &tiles_s.queue[queue_idx].a[tile_start_m][wgmma_start_k],
                        &tiles_s.queue[queue_idx].b[0][wgmma_start_k]
                    );
                }

                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(0): "memory");
                if (consumer_thread_id < CLUSTER_SIZE)
                {
                    // mark the queue slot as empty after consuming the data,
                    // for all blocks in the cluster
                    ptx_arrive_barrier_across_cluster(
                        &queue_empty[queue_idx],
                        consumer_thread_id,
                        1
                    );
                }
            }
            {
                // first two lanes store the smem locations for the warp
                asm volatile("cp.async.bulk.wait_group.read 0;\n");
                if (thread_lane_idx_n <= 1)
                {
                    // lane 0 stores nnz for the first 8 x 8 quadrant,
                    // lane 1 for the second quadrant
                    tiles_s.c_packed[thread_store_offset_m + 8*thread_lane_idx_n][0] = 0;
                }
                __syncwarp();
                
                // within each 16 x T_n/16 slice, each thread processes
                // 4 * 2 elements in each 8 x 8 quadrant
                #pragma unroll
                for(int quadrant_slice_m = 0; quadrant_slice_m < 4; quadrant_slice_m+=2)
                {
                    int quadrant_store_offset_m =
                        thread_store_offset_m + quadrant_slice_m * 4;
                    #pragma unroll
                    for(int wgmma_slice_n = 0; wgmma_slice_n < T_n / 16; wgmma_slice_n+=1)
                    {
                        int quadrant_store_offset_n =
                            thread_store_offset_n + wgmma_slice_n * 16;
                        #pragma unroll
                        for(int quadrant_slice_n = 0; quadrant_slice_n < 8; quadrant_slice_n+=4)
                        {
                            #pragma unroll
                            for(int element_n = 0; element_n < 2; element_n++)
                            {
                                if (C_accum[wgmma_slice_n][quadrant_slice_m + quadrant_slice_n + element_n] > 0)
                                {
                                    // fast SM90+ atomic
                                    uint current_store_idx = __nv_atomic_fetch_add(
                                        &tiles_s.c_packed[quadrant_store_offset_m][0],
                                        1u,
                                        __NV_ATOMIC_RELAXED,
                                        __NV_THREAD_SCOPE_BLOCK
                                    );
                                    // store the nonzero index and value packed
                                    if constexpr (LOOP_OVERFLOW_STORAGE)
                                    {
                                        tiles_s.c_packed[quadrant_store_offset_m]
                                                        [(current_store_idx & (T_n_compressed - 1)) + 1] =
                                        tile_coord_n * T_n + quadrant_store_offset_n + quadrant_slice_n * 2 + element_n |
                                        (
                                            static_cast<uint32_t>(
                                            __bfloat16_as_ushort(__float2bfloat16(
                                                C_accum[wgmma_slice_n][quadrant_slice_m + quadrant_slice_n + element_n]
                                            )
                                            )
                                        ) << 16
                                        );
                                    }
                                    else
                                    {
                                        tiles_s.c_packed[quadrant_store_offset_m][current_store_idx + 1] =
                                        tile_coord_n * T_n + quadrant_store_offset_n + quadrant_slice_n * 2 + element_n |
                                        (
                                            static_cast<uint32_t>(
                                            __bfloat16_as_ushort(__float2bfloat16(
                                                C_accum[wgmma_slice_n][quadrant_slice_m + quadrant_slice_n + element_n]
                                            )
                                            )
                                        ) << 16
                                        );
                                    }
                                }
                            } 
                        }

                    }
                }
                asm volatile("fence.proxy.async.shared::cta;\n");
                // wait for both consumer warpgroups in the cluster to finish
                // writing from registers -> smem
                asm volatile("bar.sync 10, 256;\n");
                // first thread of the first consumer warpgroup issues the TMA
                // async store
                if (threadIdx.x == 128)
                {
                    ptx_store_transposed_tile_tma_3d<uint32_t, T_n_compressed>(
                        &C_packed_tm,
                        // source tile from smem
                        &tiles_s.c_packed[0][0],
                        // start in dimensions
                        tile_coord_m * T_m,
                        tile_coord_n * T_n_compressed
                    );
                    asm volatile("cp.async.bulk.commit_group;\n");
                }
            }
            
        }
    }

}


// encodes a 2D bf16 tensor map for tiled TMA loads.
// inputs: output tensor-map storage, base pointer, logical dims. output: tensor_map filled.
template <const int T_dim0, const int T_dim1, const bool SWIZZLE = true>
void create_tensor_map_2d(
    CUtensorMap* tensor_map, __nv_bfloat16* d_ptr,
    const int dim0, const int dim1)
{
    cuuint64_t dimensions[2] = {
        static_cast<cuuint64_t>(dim1), static_cast<cuuint64_t>(dim0)};
    cuuint64_t strides[1] = {
        static_cast<cuuint64_t>(dim1) * sizeof(__nv_bfloat16)};
    cuuint32_t box_dimensions[2] = {
        static_cast<cuuint32_t>(T_dim1), static_cast<cuuint32_t>(T_dim0)};
    cuuint32_t element_traversal_strides[2] = {1, 1};

    cuTensorMapEncodeTiled(
        tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        static_cast<void*>(d_ptr),
        dimensions,
        strides,
        box_dimensions,
        element_traversal_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        SWIZZLE ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// encodes the transposed 3D uint32 tensor map used by packed-C TMA stores.
// inputs: output tensor-map storage, base pointer, logical dims. output: tensor_map filled.
template <const int T_dim0, const int T_dim1, const bool SWIZZLE = true,
          int last_dim_size = 64, int PADDING = 0>
void create_transposed_tensor_map_uint32_3d(
    CUtensorMap* tensor_map, uint32_t* d_ptr,
    const int dim0, const int dim1)
{
    static_assert(
        T_dim1 % last_dim_size == 0,
        "Optimized only when the last tile dimension is divisible by last_dim_size");

    cuuint64_t dimensions[3] = {
        static_cast<cuuint64_t>(last_dim_size),
        static_cast<cuuint64_t>(dim0),
        static_cast<cuuint64_t>(dim1 / last_dim_size)};
    cuuint64_t strides[2] = {
        static_cast<cuuint64_t>(dim1) * sizeof(uint32_t),
        last_dim_size * sizeof(uint32_t)};
    cuuint32_t box_dimensions[3] = {
        static_cast<cuuint32_t>(last_dim_size + PADDING),
        static_cast<cuuint32_t>(T_dim0),
        static_cast<cuuint32_t>(T_dim1 / last_dim_size)};
    cuuint32_t element_traversal_strides[3] = {1, 1, 1};

    cuTensorMapEncodeTiled(
        tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_UINT32,
        3,
        static_cast<void*>(d_ptr),
        dimensions,
        strides,
        box_dimensions,
        element_traversal_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        SWIZZLE ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}


struct D2TLayerCache {
    CUtensorMap A_tm{};
    CUtensorMap B_tm{};
    CUtensorMap C_packed_tm{};
    __nv_bfloat16* A_ptr = nullptr;
    __nv_bfloat16* B_ptr = nullptr;
    uint32_t* C_packed_ptr = nullptr;
    bool runtime_maps_initialized = false;
    size_t M = 0;
    int K = -1;
    int N = -1;
    int T_m = -1;
    int T_n = -1;
    int T_k = -1;
    int T_n_compressed = -1;
};

struct D2TSharedScheduleCache {
    int* partitioned_schedule_d = nullptr;
    int schedule_size_per_sm = -1;
    int schedule_num_clusters_m = -1;
    int schedule_num_clusters_n = -1;
    int schedule_num_active_clusters = -1;
};

static std::unordered_map<int, D2TLayerCache> CACHED_D2T_LAYERS;
static D2TSharedScheduleCache SHARED_D2T_SCHEDULE;

// releases the cached per-shape schedule buffer shared across D2T layers.
// inputs: none. output: schedule cache reset and device memory freed if needed.
static void release_shared_schedule_cache() {
    if (SHARED_D2T_SCHEDULE.partitioned_schedule_d != nullptr) {
        cudaError_t status = cudaFree(SHARED_D2T_SCHEDULE.partitioned_schedule_d);
        if (status != cudaSuccess) {
            printf("cudaFree for cached schedule failed: %s\n", cudaGetErrorString(status));
        }
        SHARED_D2T_SCHEDULE.partitioned_schedule_d = nullptr;
    }
    SHARED_D2T_SCHEDULE.schedule_size_per_sm = -1;
    SHARED_D2T_SCHEDULE.schedule_num_clusters_m = -1;
    SHARED_D2T_SCHEDULE.schedule_num_clusters_n = -1;
    SHARED_D2T_SCHEDULE.schedule_num_active_clusters = -1;
}

// initializes the static cache for one D2T layer configuration.
// inputs: layer cache, B pointer, logical K/N. output: B tensor map and metadata refreshed.
template <
    const int T_m,
    const int T_n,
    const int T_k,
    const int CLUSTER_DIM_m,
    const int CLUSTER_DIM_n,
    const int QUEUE_SIZE,
    const int NUM_ACTIVE_SMs = NUM_SMs,
    const int T_n_compressed = T_n / 8
>
void prepare_d2t_layer_static_cache(
    D2TLayerCache& cache,
    at::BFloat16* B_d_at,
    const int K, const int N
)
{
    __nv_bfloat16* B_d = reinterpret_cast<__nv_bfloat16*>(B_d_at);
    static_assert(
        (T_k == 64),
        "Only T_m == 128 supported"
    );

    static_assert(
        (T_m == 64 * 2),
        "Only T_m == 128 supported"
    );
    static_assert(
        (T_n == 64 * 4),
        "Only T_n == 256 supported"
    );
    static_assert(
        (NUM_ACTIVE_SMs >= 1 && NUM_ACTIVE_SMs <= NUM_SMs),
        "NUM_ACTIVE_SMs must be in [1, NUM_SMs]"
    );
    static_assert(
        (NUM_ACTIVE_SMs % (CLUSTER_DIM_m * CLUSTER_DIM_n) == 0),
        "NUM_ACTIVE_SMs must be divisible by the number of blocks in a cluster "
        "(CLUSTER_DIM_m * CLUSTER_DIM_n)"
    );
    static_assert(
        (CLUSTER_DIM_m * CLUSTER_DIM_n <= 8),
        "Cluster size cannot exceed 8 (CUDA limit)"
    );
    assert((N / T_n) % CLUSTER_DIM_n == 0);

    if (cache.B_ptr != B_d ||
        cache.N != N || cache.K != K ||
        cache.T_m != T_m || cache.T_n != T_n || cache.T_k != T_k ||
        cache.T_n_compressed != T_n_compressed)
    {
        create_tensor_map_2d<T_n, T_k>(
            &cache.B_tm,
            B_d,
            N,
            K
        );
        cache.B_ptr = B_d;
        cache.M = 0;
        cache.N = N;
        cache.K = K;
        cache.T_m = T_m;
        cache.T_n = T_n;
        cache.T_k = T_k;
        cache.T_n_compressed = T_n_compressed;
        cache.A_ptr = nullptr;
        cache.C_packed_ptr = nullptr;
        cache.runtime_maps_initialized = false;
    }
}

// updates runtime tensor-map addresses for the current A and packed-C buffers.
// inputs: layer cache, A pointer, packed-C pointer. output: runtime tensor maps ready to launch.
template <
    const int T_m,
    const int T_n,
    const int T_k,
    const int CLUSTER_DIM_m,
    const int CLUSTER_DIM_n,
    const int QUEUE_SIZE,
    const int NUM_ACTIVE_SMs = NUM_SMs,
    const int T_n_compressed = T_n / 8
>
void update_d2t_layer_runtime_maps(
    D2TLayerCache& cache,
    at::BFloat16* A_d_at,
    uint32_t* C_packed_d
)
{
    __nv_bfloat16* A_d = reinterpret_cast<__nv_bfloat16*>(A_d_at);
    CUresult result;
    if (!cache.runtime_maps_initialized) {
        create_tensor_map_2d<T_m, T_k>(
            &cache.A_tm,
            A_d,
            cache.M,
            cache.K
        );
        create_transposed_tensor_map_uint32_3d<T_m, T_n_compressed, false, T_n_compressed, 4>(
            &cache.C_packed_tm,
            C_packed_d,
            cache.M,
            cache.N / T_n * T_n_compressed
        );
        cache.runtime_maps_initialized = true;
    } else {
        if (cache.A_ptr != A_d) {
            result = cuTensorMapReplaceAddress(&cache.A_tm, A_d);
            if (result != CUDA_SUCCESS) {
                printf("cuTensorMapReplaceAddress(A) failed: %d\n", (int)result);
                exit(-1);
            }
        }
        if (cache.C_packed_ptr != C_packed_d) {
            result = cuTensorMapReplaceAddress(&cache.C_packed_tm, C_packed_d);
            if (result != CUDA_SUCCESS) {
                printf("cuTensorMapReplaceAddress(C_packed) failed: %d\n", (int)result);
                exit(-1);
            }
        }
    }
    cache.A_ptr = A_d;
    cache.C_packed_ptr = C_packed_d;
}

// ensures the cached schedule and shape-dependent state match the requested M.
// inputs: layer cache, logical M. output: schedule cache and runtime invalidation updated.
template <
    const int T_m,
    const int T_n,
    const int T_k,
    const int CLUSTER_DIM_m,
    const int CLUSTER_DIM_n,
    const int QUEUE_SIZE,
    const int NUM_ACTIVE_SMs = NUM_SMs,
    const int T_n_compressed = T_n / 8
>
void ensure_d2t_layer_shape(
    D2TLayerCache& cache,
    const size_t M
)
{
    constexpr int CLUSTER_SIZE = CLUSTER_DIM_m * CLUSTER_DIM_n;
    constexpr int NUM_ACTIVE_CLUSTERS = NUM_ACTIVE_SMs / CLUSTER_SIZE;
    const int num_clusters_m = (M + T_m - 1) / (T_m * CLUSTER_DIM_m);
    const int num_clusters_n = (cache.N + T_n - 1) / (T_n * CLUSTER_DIM_n);
    const int schedule_size_per_sm =
        (num_clusters_m * num_clusters_n + NUM_ACTIVE_CLUSTERS - 1)
        / NUM_ACTIVE_CLUSTERS;
    assert((M / T_m) % CLUSTER_DIM_m == 0);
    const bool reuse_shared_schedule =
        SHARED_D2T_SCHEDULE.partitioned_schedule_d != nullptr &&
        SHARED_D2T_SCHEDULE.schedule_size_per_sm == schedule_size_per_sm &&
        SHARED_D2T_SCHEDULE.schedule_num_clusters_m == num_clusters_m &&
        SHARED_D2T_SCHEDULE.schedule_num_clusters_n == num_clusters_n &&
        SHARED_D2T_SCHEDULE.schedule_num_active_clusters == NUM_ACTIVE_CLUSTERS;

    if (!reuse_shared_schedule) {
        const size_t partitioned_schedule_total_size =
            static_cast<size_t>(NUM_ACTIVE_CLUSTERS) *
            static_cast<size_t>(schedule_size_per_sm);
        const size_t partitioned_schedule_bytes =
            partitioned_schedule_total_size * sizeof(int);
        std::vector<int> schedule_h(partitioned_schedule_total_size);
        build_partitioned_hilbert_schedule_for_grid(
            num_clusters_m,
            num_clusters_n,
            NUM_ACTIVE_CLUSTERS,
            schedule_size_per_sm,
            schedule_h.data()
        );

        release_shared_schedule_cache();
        cudaError_t status = cudaMalloc(
            reinterpret_cast<void**>(&SHARED_D2T_SCHEDULE.partitioned_schedule_d),
            partitioned_schedule_bytes
        );
        if (status != cudaSuccess) {
            printf("cudaMalloc for schedule failed: %s\n", cudaGetErrorString(status));
            exit(-1);
        }

        status = cudaMemcpy(
            SHARED_D2T_SCHEDULE.partitioned_schedule_d,
            schedule_h.data(),
            partitioned_schedule_bytes,
            cudaMemcpyHostToDevice
        );
        if (status != cudaSuccess) {
            printf("cudaMemcpy for schedule failed: %s\n", cudaGetErrorString(status));
            exit(-1);
        }

        SHARED_D2T_SCHEDULE.schedule_size_per_sm = schedule_size_per_sm;
        SHARED_D2T_SCHEDULE.schedule_num_clusters_m = num_clusters_m;
        SHARED_D2T_SCHEDULE.schedule_num_clusters_n = num_clusters_n;
        SHARED_D2T_SCHEDULE.schedule_num_active_clusters = NUM_ACTIVE_CLUSTERS;
    }

    if (cache.M != M) {
        cache.M = M;
        cache.A_ptr = nullptr;
        cache.C_packed_ptr = nullptr;
        cache.runtime_maps_initialized = false;
    }
}

// launches the cached D2T kernel for the current layer state.
// inputs: fully prepared layer cache. output: kernel dispatched on the current CUDA stream.
template <
    const int T_m,
    const int T_n,
    const int T_k,
    const int CLUSTER_DIM_m,
    const int CLUSTER_DIM_n,
    const int QUEUE_SIZE,
    const int NUM_ACTIVE_SMs = NUM_SMs,
    const int T_n_compressed = T_n / 8,
    const bool LOOP_OVERFLOW_STORAGE
>
void run_d2t_layer_cache(
    const D2TLayerCache& cache,
    cudaStream_t stream = 0
)
{
    constexpr int BASE_SMEM_SIZE = sizeof(SmemStorage<T_m, T_n, T_k, QUEUE_SIZE, T_n_compressed>);
    const int SMEM_SIZE = BASE_SMEM_SIZE + SHARED_D2T_SCHEDULE.schedule_size_per_sm * sizeof(int);
    auto* kernel =
        mm_wgmma_nt_kernel<T_m, T_n, T_k, CLUSTER_DIM_m, CLUSTER_DIM_n,
                           QUEUE_SIZE, NUM_ACTIVE_SMs, T_n_compressed, LOOP_OVERFLOW_STORAGE>;
    cudaError_t status = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_SIZE);

    if (status != cudaSuccess) {
        printf("cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(status));
        exit(-1);
    }

    dim3 grid_dim(NUM_ACTIVE_SMs);
    dim3 block_dim(NUM_THREADS_PER_BLOCK);
    kernel<<<grid_dim, block_dim, SMEM_SIZE, stream>>>(
            cache.A_tm, cache.B_tm,
            cache.C_packed_tm,
            SHARED_D2T_SCHEDULE.partitioned_schedule_d, SHARED_D2T_SCHEDULE.schedule_size_per_sm,
            cache.K
        );
}

// creates or refreshes the cached 128x256x64 TS8 D2T layer metadata.
// inputs: layer id, B pointer, logical K/N. output: static layer cache prepared.
void create_d2t_layer_128x256x64TS8(
    const int layer_number,
    at::BFloat16* B_d,
    const int K, const int N
) {
    auto& layer_cache = CACHED_D2T_LAYERS[layer_number];
    prepare_d2t_layer_static_cache<128, 256, 64, 2, 1, 4, 128, 32>(
        layer_cache,
        B_d,
        K, N
    );
}

// creates or refreshes the cached 128x256x64 TS4 D2T layer metadata.
// inputs: layer id, B pointer, logical K/N. output: static layer cache prepared.
void create_d2t_layer_128x256x64TS4(
    const int layer_number,
    at::BFloat16* B_d,
    const int K, const int N
) {
    auto& layer_cache = CACHED_D2T_LAYERS[layer_number];
    prepare_d2t_layer_static_cache<128, 256, 64, 2, 1, 3, 128, 64>(
        layer_cache,
        B_d,
        K, N
    );
}

// creates or refreshes the cached 128x256x64 TS2 D2T layer metadata.
// inputs: layer id, B pointer, logical K/N. output: static layer cache prepared.
void create_d2t_layer_128x256x64TS2(
    const int layer_number,
    at::BFloat16* B_d,
    const int K, const int N
) {
    auto& layer_cache = CACHED_D2T_LAYERS[layer_number];
    prepare_d2t_layer_static_cache<128, 256, 64, 2, 1, 3, 128, 128>(
        layer_cache,
        B_d,
        K, N
    );
}

// ensures the cached 128x256x64 TS8 layer shape matches the requested M.
// inputs: layer id, logical M. output: shape-dependent schedule state prepared.
void ensure_d2t_layer_shape_128x256x64TS8(
    const int layer_number,
    const size_t M
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("ensure_d2t_layer_shape called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    ensure_d2t_layer_shape<128, 256, 64, 2, 1, 4, 128, 32>(
        it->second,
        M
    );
}

// ensures the cached 128x256x64 TS4 layer shape matches the requested M.
// inputs: layer id, logical M. output: shape-dependent schedule state prepared.
void ensure_d2t_layer_shape_128x256x64TS4(
    const int layer_number,
    const size_t M
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("ensure_d2t_layer_shape called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    ensure_d2t_layer_shape<128, 256, 64, 2, 1, 3, 128, 64>(
        it->second,
        M
    );
}

// ensures the cached 128x256x64 TS2 layer shape matches the requested M.
// inputs: layer id, logical M. output: shape-dependent schedule state prepared.
void ensure_d2t_layer_shape_128x256x64TS2(
    const int layer_number,
    const size_t M
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("ensure_d2t_layer_shape called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    ensure_d2t_layer_shape<128, 256, 64, 2, 1, 3, 128, 128>(
        it->second,
        M
    );
}

// runs the cached 128x256x64 TS8 D2T layer for one input/output pair.
// inputs: layer id, A pointer, packed-C pointer. output: packed positive activations written.
void run_d2t_layer_128x256x64TS8(
    const int layer_number,
    at::BFloat16* A_d,
    uint32_t* C_packed_d,
    cudaStream_t stream = 0
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("run_d2t_layer called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    update_d2t_layer_runtime_maps<128, 256, 64, 2, 1, 4, 128, 32>(
        it->second,
        A_d,
        C_packed_d
    );
    run_d2t_layer_cache<128, 256, 64, 2, 1, 4, 128, 32, true>(
        it->second,
        stream
    );
}

// runs the cached 128x256x64 TS4 D2T layer for one input/output pair.
// inputs: layer id, A pointer, packed-C pointer. output: packed positive activations written.
void run_d2t_layer_128x256x64TS4(
    const int layer_number,
    at::BFloat16* A_d,
    uint32_t* C_packed_d,
    cudaStream_t stream = 0
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("run_d2t_layer called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    update_d2t_layer_runtime_maps<128, 256, 64, 2, 1, 3, 128, 64>(
        it->second,
        A_d,
        C_packed_d
    );
    run_d2t_layer_cache<128, 256, 64, 2, 1, 3, 128, 64, true>(
        it->second,
        stream
    );
}

// runs the cached 128x256x64 TS2 D2T layer for one input/output pair.
// inputs: layer id, A pointer, packed-C pointer. output: packed positive activations written.
void run_d2t_layer_128x256x64TS2(
    const int layer_number,
    at::BFloat16* A_d,
    uint32_t* C_packed_d,
    cudaStream_t stream = 0
) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        printf("run_d2t_layer called for non-existing layer %d\n", layer_number);
        exit(-1);
    }
    update_d2t_layer_runtime_maps<128, 256, 64, 2, 1, 3, 128, 128>(
        it->second,
        A_d,
        C_packed_d
    );
    run_d2t_layer_cache<128, 256, 64, 2, 1, 3, 128, 128, true>(
        it->second,
        stream
    );
}


// destroys one cached D2T layer entry.
// inputs: layer id. output: layer cache removed and shared schedule freed if now unused.
void destroy_d2t_layer(const int layer_number) {
    auto it = CACHED_D2T_LAYERS.find(layer_number);
    if (it == CACHED_D2T_LAYERS.end()) {
        return;
    }
    CACHED_D2T_LAYERS.erase(it);
    if (CACHED_D2T_LAYERS.empty()) {
        release_shared_schedule_cache();
    }
}

// destroys all cached D2T layer entries and the shared schedule cache.
// inputs: none. output: all cached D2T state released.
void destroy_all_d2t_layers() {
    CACHED_D2T_LAYERS.clear();
    release_shared_schedule_cache();
}

// runs the 128x256x64 TS8 D2T path one-shot without explicit cache management.
// inputs: A pointer, B pointer, packed-C pointer, logical M/K/N. output: packed positive activations written.
void mm_wgmma_nt_128x256x64TS8(
    at::BFloat16* A_d, at::BFloat16* B_d, uint32_t* C_packed_d,
    const int M, const int K, const int N,
    cudaStream_t stream
) {
    constexpr int LEGACY_LAYER_ID = -1;
    create_d2t_layer_128x256x64TS8(
        LEGACY_LAYER_ID,
        B_d,
        K, N
    );
    ensure_d2t_layer_shape_128x256x64TS8(
        LEGACY_LAYER_ID,
        M
    );
    run_d2t_layer_128x256x64TS8(
        LEGACY_LAYER_ID,
        A_d,
        C_packed_d,
        stream
    );
}

// runs the 128x256x64 TS4 D2T path one-shot without explicit cache management.
// inputs: A pointer, B pointer, packed-C pointer, logical M/K/N. output: packed positive activations written.
void mm_wgmma_nt_128x256x64TS4(
    at::BFloat16* A_d, at::BFloat16* B_d, uint32_t* C_packed_d,
    const int M, const int K, const int N,
    cudaStream_t stream
) {
    constexpr int LEGACY_LAYER_ID = -1;
    create_d2t_layer_128x256x64TS4(
        LEGACY_LAYER_ID,
        B_d,
        K, N
    );
    ensure_d2t_layer_shape_128x256x64TS4(
        LEGACY_LAYER_ID,
        M
    );
    run_d2t_layer_128x256x64TS4(
        LEGACY_LAYER_ID,
        A_d,
        C_packed_d,
        stream
    );
}

// runs the 128x256x64 TS2 D2T path one-shot without explicit cache management.
// inputs: A pointer, B pointer, packed-C pointer, logical M/K/N. output: packed positive activations written.
void mm_wgmma_nt_128x256x64TS2(
    at::BFloat16* A_d, at::BFloat16* B_d, uint32_t* C_packed_d,
    const int M, const int K, const int N,
    cudaStream_t stream
) {
    constexpr int LEGACY_LAYER_ID = -1;
    create_d2t_layer_128x256x64TS2(
        LEGACY_LAYER_ID,
        B_d,
        K, N
    );
    ensure_d2t_layer_shape_128x256x64TS2(
        LEGACY_LAYER_ID,
        M
    );
    run_d2t_layer_128x256x64TS2(
        LEGACY_LAYER_ID,
        A_d,
        C_packed_d,
        stream
    );
}

}

#undef WARP_SIZE
#undef WARP_GROUP_SIZE
#undef NUM_THREADS_PER_BLOCK
#undef NUM_SMs
#undef WGMMA_m
#undef WGMMA_k
