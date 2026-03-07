#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

namespace cub = hipcub;

#define GPU_CHECK(ans)                                                                   \
    {                                                                                    \
        gpuAssert((ans), __FILE__, __LINE__);                                            \
    }
inline void
gpuAssert(hipError_t code, const char* file, int line, bool abort = true)
{
  if(code != hipSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

template <typename scalar_t>
__device__ void _moe_align_block_size(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts,
    int padded_num_experts, int experts_per_warp, int block_size,
    int numel, int* __restrict__ cumsum, int max_num_tokens_padded,
    int max_num_m_blocks, int model_offset, int inactive_expert_id,
    int topk_num, int* token_mask)
{
  extern __shared__ int shared_counts[];

  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;
  int cumsum_offset = (num_experts + 1) * model_offset;

  // Use separate threadblocks to fill sorted_token_ids.
  // This is safe since the current kernel does not use sorted_token_ids.
  if (blockIdx.x % 2) {
    // Initialize sorted_token_ids with numel
    for (int it = threadIdx.x; it < max_num_tokens_padded;
         it += blockDim.x) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    return;
  }

  const int warp_id = threadIdx.x / warpSize;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset],
              mask);
  }

  __syncthreads();

  // Compute prefix sum over token counts per expert
  using BlockScan = cub::BlockScan<int, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[cumsum_offset + expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    total_tokens_post_pad[model_offset] = cumsum_val;
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[cumsum_offset + threadIdx.x];
         i < cumsum[cumsum_offset + threadIdx.x + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = threadIdx.x;
    }
  }

  // Fill remaining expert_ids with 0
  const int fill_start_idx =
      cumsum[cumsum_offset + num_experts] / block_size + threadIdx.x;
  for (int i = fill_start_idx; i < max_num_m_blocks; i += blockDim.x) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }
}

template <typename scalar_t, int fill_threads>
__device__ void _moe_align_block_size_small_batch_expert(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts, int block_size,
    int numel, int max_num_tokens_padded, int max_num_m_blocks,
    int inactive_expert_id, int model_offset, int topk_num,
    int* token_mask)
{
  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;

  // Use an additional group of threads to fill sorted_token_ids.
  // Since the current kernel will use sorted_token_ids afterward,
  // we fill sorted_token_ids within the same threadblock to make
  // synchronization easier.
  if (threadIdx.x < fill_threads) {
    // Initialize sorted_token_ids with numel
    for (int it = threadIdx.x; it < max_num_tokens_padded;
         it += fill_threads) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    // Three __syncthreads() corresponding to the other threads
    __syncthreads();
    //__syncthreads();
    //__syncthreads();
    return;
  }

  const int tid = threadIdx.x - fill_threads;
  const int stride = blockDim.x - fill_threads;

  extern __shared__ int shared_mem[];
  int* cumsum = shared_mem;
  int* tokens_cnts = (int*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    tokens_cnts[(tid + 1) * num_experts + expert_id] += mask;
  }

  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
          CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) * block_size;
    }
    total_tokens_post_pad[model_offset] =
        static_cast<int>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = tid;
    }
  }

  // Fill remaining expert_ids with 0
  const int fill_start_idx = cumsum[num_experts] / block_size + tid;
  for (int i = fill_start_idx; i < max_num_m_blocks; i += stride) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int rank_post_pad =
        tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      sorted_token_ids[sorted_token_ids_offset + rank_post_pad] = i;
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }
}

template <typename scalar_t>
__device__ void _count_and_sort_expert_tokens(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ cumsum_buffer,
    int* __restrict__ expert_map, int numel, int num_experts,
    int max_num_tokens_padded, int* __restrict__ token_mask,
    int model_offset, int topk_num)
{
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.y;

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }

    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      int rank_post_pad = atomicAdd(&cumsum_buffer[(model_offset * (num_experts + 1)) + expert_id], 1);
      sorted_token_ids[max_num_tokens_padded * model_offset + rank_post_pad] = i;
    }
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts,
    int padded_num_experts, int experts_per_warp, int block_size,
    int numel, int* __restrict__ cumsum, int max_num_tokens_padded,
    int topk_num)
{
  _moe_align_block_size(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, padded_num_experts, experts_per_warp, block_size, numel,
      cumsum, max_num_tokens_padded, CEILDIV(max_num_tokens_padded, block_size),
      0, 0, topk_num, nullptr);
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ cumsum_buffer,
    int* __restrict__ expert_map, int numel, int num_experts,
    int max_num_tokens_padded, int topk_num)
{
  _count_and_sort_expert_tokens(
      topk_ids, sorted_token_ids, cumsum_buffer, expert_map, numel, num_experts,
      max_num_tokens_padded, nullptr, 0, topk_num);
}

template <typename scalar_t, int fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts, int block_size,
    int numel, int max_num_tokens_padded, int topk_num)
{

  _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, block_size, numel, max_num_tokens_padded,
      CEILDIV(max_num_tokens_padded, block_size), 0, 0, topk_num, nullptr);
}


// taken from
// https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc
void moe_align_block_size(int *topk_ids, int num_experts,
                          int block_size, int *sorted_token_ids,
                          int *experts_ids,
                          int *num_tokens_post_pad,
                          int *expert_map,
                          int topk_ids_size,
                          int topk,
                          int sorted_token_ids_size)
{
  int WARP_SIZE;
  GPU_CHECK(hipDeviceGetAttribute(&WARP_SIZE, hipDeviceAttributeWarpSize, 0));

  int threads = 1024;

  int64_t padded_num_experts = ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  assert(padded_num_experts < 1024);

  // calc needed amount of shared mem for `cumsum` tensors
  bool small_batch_expert_mode = (topk_ids_size < 1024) && (num_experts <= 64);

  if (small_batch_expert_mode) {
    const int threads = std::max(num_experts, WARP_SIZE);
    const int shared_mem_size =
        ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int);

    // threadIdx.x >= fill_threads: counting experts and aligning
    // threadIdx.x < fill_threads: filling sorted_token_ids
    constexpr int fill_threads = 256;
    auto small_batch_expert_kernel = moe_align_block_size_small_batch_expert_kernel<int, fill_threads>;
    small_batch_expert_kernel<<<1, fill_threads + threads, shared_mem_size, 0>>>(
        topk_ids,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map, num_experts, block_size,
        topk_ids_size, sorted_token_ids_size, topk);
    GPU_CHECK(hipDeviceSynchronize());
  } else {
    int *cumsum_buff;
    GPU_CHECK(hipMalloc((void**)&cumsum_buff, sizeof(int) * (num_experts + 1)));
    GPU_CHECK(hipMemset(cumsum_buff, 0, sizeof(int) * (num_experts + 1)));

    auto align_kernel = moe_align_block_size_kernel<int>;

    int num_warps = CEILDIV(padded_num_experts, experts_per_warp);
    int shared_mem_size = num_warps * experts_per_warp * sizeof(int);

    // launch two threadblocks
    // blockIdx.x == 0: counting experts and aligning
    // blockIdx.x == 1: filling sorted_token_ids
    align_kernel<<<2, threads, shared_mem_size, 0>>>(
        topk_ids,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map, num_experts, padded_num_experts,
        experts_per_warp, block_size, topk_ids_size,
        cumsum_buff, sorted_token_ids_size,
        topk);

    const int block_threads = std::min(256, (int)threads);
    const int num_blocks = (topk_ids_size + block_threads - 1) / block_threads;
    const int max_blocks = 65535;
    const int actual_blocks = std::min(num_blocks, max_blocks);
    dim3 gridDims(1, actual_blocks);

    auto sort_kernel = count_and_sort_expert_tokens_kernel<int>;
    sort_kernel<<<gridDims, block_threads>>>(
        topk_ids,
        sorted_token_ids,
        cumsum_buff, expert_map,
        topk_ids_size, num_experts, sorted_token_ids_size,
        topk);

    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipFree(cumsum_buff));
  }
}

/*
def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.
    - pad_sorted_ids: A flag indicating whether the sorted_token_ids length
        should be padded to a multiple of block_size,
    - ignore_invalid_experts: A flag indicating whether to ignore invalid
        experts. When False, all expert_ids in topk_ids will participate in
        counting and ranking, but invalid experts in expert_ids will be marked
        as -1. When True, all invalid expert_ids in topk_ids will be ignored
        and will not participate in counting or ranking, and there will be no
        -1 in expert_ids.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad
*/

void randperm(int* topk_ids, int topk, int m, int n) {
  std::vector<int> v(n);
  for (int i = 0; i < n; ++i) v[i] = i;
  std::mt19937 gen(19937);
  for (int i = 0; i < m; i++) { 
    std::shuffle(v.begin(), v.end(), gen);
    for (int j = 0; j < topk; j++) {
      topk_ids[i * topk + j] = v[j];
      #ifdef DEBUG
      printf("%d ", v[j]);    
      #endif
    }
    #ifdef DEBUG
    printf("\n");
    #endif
  }
}


int main(int argc, char* argv[])
{
   if (argc != 2) {
     printf("Usage: %s <repeat>\n", argv[0]);
     return 1;
   }
   const int repeat = atoi(argv[1]);

#ifdef DEBUG
   int tokens[] = {4};
   int experts[] = {4};
   int topks[] = {3};
   int block_sizes[] = {4};
#else
   int tokens[] = {1, 3, 256, 4096, 8192};
   int experts[] = {32, 128};
   int topks[] = {2, 3, 4};
   int block_sizes[] = {32};
#endif
   bool pad_sorted_ids = true;

   int *topk_ids, *h_topk_ids;

   for (int i = 0; i < int(sizeof(tokens) / sizeof(tokens[0])); i++) {
     int m = tokens[i]; 

     for (int e = 0; e < int(sizeof(experts) / sizeof(experts[0])); e++) {
       int num_experts = experts[e]; 

       for (int k = 0; k < int(sizeof(topks) / sizeof(topks[0])); k++) {
         int topk = topks[k]; 
         int block_size = block_sizes[0];

         int topk_ids_size = m * topk;
         h_topk_ids = (int*) malloc(topk_ids_size * sizeof(int));
         randperm(h_topk_ids, topk, m, num_experts);

         GPU_CHECK(hipMalloc((void**)&topk_ids, m * topk * sizeof(int)));
         GPU_CHECK(hipMemcpy(topk_ids, h_topk_ids, topk_ids_size * sizeof(int), hipMemcpyHostToDevice));
         
         int max_num_tokens_padded = topk_ids_size + num_experts * (block_size - 1);

         if (pad_sorted_ids)
             max_num_tokens_padded = CEILDIV(max_num_tokens_padded, block_size) * block_size;

         if (topk_ids_size < num_experts)
             max_num_tokens_padded = std::min(topk_ids_size * block_size, max_num_tokens_padded);

         #ifdef DEBUG
         printf("max_num_tokens_padded: %zu\n",  max_num_tokens_padded);
         #endif

         int *sorted_ids;
         GPU_CHECK(hipMalloc((void**)&sorted_ids, max_num_tokens_padded * sizeof(int)));

         int *expert_ids;
         int max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);
         GPU_CHECK(hipMalloc((void**)&expert_ids, max_num_m_blocks * sizeof(int)));

         int *num_tokens_post_pad;
         GPU_CHECK(hipMalloc((void**)&num_tokens_post_pad, sizeof(int)));

         //bool ignore_invalid_experts = false;

         auto start = std::chrono::steady_clock::now();

         for (int n = 0; n < repeat; n++) {
           moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                nullptr, //expert_map if ignore_invalid_experts else None,
                topk_ids_size,
                topk,
                max_num_tokens_padded 
           );
         }

         auto end = std::chrono::steady_clock::now();
         auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
         printf("Average execution time of the kernels (tokens %d, topk: %d, expert: %d, block_size %d): %f (us)\n",
                m, topk, num_experts, block_size, (time * 1e-3f) / repeat);

         //if expert_map is not None and not ignore_invalid_experts:
             //expert_ids = expert_map[expert_ids]

         //sorted_ids, expert_ids, num_tokens_post_pad
         int actual_num_tokens;
         GPU_CHECK(hipMemcpy(&actual_num_tokens, num_tokens_post_pad, sizeof(int), hipMemcpyDeviceToHost));

         int *actual_expert_ids = (int*) malloc(max_num_m_blocks * sizeof(int));
         GPU_CHECK(hipMemcpy(actual_expert_ids, expert_ids, max_num_m_blocks * sizeof(int), hipMemcpyDeviceToHost));

         int *actual_sorted_ids;
         actual_sorted_ids = (int*) malloc (max_num_tokens_padded * sizeof(int));
         GPU_CHECK(hipMemcpy(actual_sorted_ids, sorted_ids, max_num_tokens_padded * sizeof(int), hipMemcpyDeviceToHost));

         #ifdef DEBUG
         printf("actual sorted ids: ");
         for (int i = 0; i < max_num_tokens_padded; i++) {
           printf("%d ", actual_sorted_ids[i]);
         }

         printf("\nactual expert ids: ");
         for (int i = 0; i < max_num_m_blocks; i++) {
           printf("%d ", actual_expert_ids[i]);
         }
         printf("\nactual number of tokens: %d\n", actual_num_tokens);
         #endif

         bool ok = true;
         if (actual_num_tokens % block_size) {
           ok = false;
           printf("Error: num_tokens_post_pad should be divisible by block_size\n");
         }
         if (actual_num_tokens < topk_ids_size) {
           ok = false;
           printf("Error: num_tokens_post_pad should be at least total_tokens\n");
         }
         for (int i = 0; i < max_num_m_blocks; i++) {
           if (actual_expert_ids[i] < 0 || actual_expert_ids[i] >= num_experts) {
             ok = false;
             printf("Error: expert_ids should contain valid expert indices\n");
             break;
           }
         }

         int ei = 0;
         for (int t = 0; t < max_num_tokens_padded; t += block_size) {
           int eid = actual_expert_ids[ei++];
           for (int b = 0; b < block_size; b++) { 
             int v = actual_sorted_ids[t+b];
             if (v == topk_ids_size) {
               continue;
             }
             if (eid != h_topk_ids[v]) {
               ok = false;
               break;
             }
           }
         }
         printf("%s\n", ok ? "PASS" : "FAIL");

         GPU_CHECK(hipFree(topk_ids));
         GPU_CHECK(hipFree(sorted_ids));
         GPU_CHECK(hipFree(expert_ids));
         GPU_CHECK(hipFree(num_tokens_post_pad));

         free(h_topk_ids);
         free(actual_sorted_ids);
         free(actual_expert_ids);
       }
     }
   }
   if (!ok) return 1;
   return 0;
}
