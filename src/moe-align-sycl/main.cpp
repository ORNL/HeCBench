#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sycl/sycl.hpp>


#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

template <typename scalar_t>
void _moe_align_block_size(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts,
    int padded_num_experts, int experts_per_warp, int block_size,
    int numel, int* __restrict__ cumsum, int max_num_tokens_padded,
    int max_num_m_blocks, int model_offset, int inactive_expert_id,
    int topk_num, int* token_mask, const sycl::nd_item<3> &item,
    int *shared_counts)
{
  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;
  int cumsum_offset = (num_experts + 1) * model_offset;

  // Use separate threadblocks to fill sorted_token_ids.
  // This is safe since the current kernel does not use sorted_token_ids.
  if (item.get_group(2) % 2) {
    // Initialize sorted_token_ids with numel
    for (int it = item.get_local_id(2); it < max_num_tokens_padded;
         it += item.get_local_range(2)) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    return;
  }

  const int warp_id = item.get_local_id(2) /
                      item.get_sub_group().get_local_range().get(0);
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  item.barrier(sycl::access::fence_space::local_space);

  const int tid = item.get_local_id(2);
  const int stride = item.get_local_range(2);

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
    
    auto ao = sycl::atomic_ref<int, sycl::memory_order::relaxed, \
                               sycl::memory_scope::work_group,\
                               sycl::access::address_space::generic_space>(
                shared_counts[warp_idx * experts_per_warp + expert_offset]);
    ao.fetch_add(mask);
  }

  item.barrier(sycl::access::fence_space::local_space);

  // Compute prefix sum over token counts per expert

  int expert_count = 0;
  int expert_id = item.get_local_id(2);
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  cumsum_val = sycl::exclusive_scan_over_group(item.get_group(),
                                               expert_count, 0, sycl::plus<>());
  if (expert_id <= num_experts) {
    cumsum[cumsum_offset + expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    total_tokens_post_pad[model_offset] = cumsum_val;
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (item.get_local_id(2) < num_experts) {
    for (int i = cumsum[cumsum_offset + item.get_local_id(2)];
         i < cumsum[cumsum_offset + item.get_local_id(2) + 1];
         i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = item.get_local_id(2);
    }
  }

  // Fill remaining expert_ids with 0
  const int fill_start_idx = cumsum[cumsum_offset + num_experts] / block_size +
                             item.get_local_id(2);
  for (int i = fill_start_idx; i < max_num_m_blocks;
       i += item.get_local_range(2)) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }
}

template <typename scalar_t, int fill_threads>
void _moe_align_block_size_small_batch_expert(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts, int block_size,
    int numel, int max_num_tokens_padded, int max_num_m_blocks,
    int inactive_expert_id, int model_offset, int topk_num,
    int* token_mask, const sycl::nd_item<3> &item, int *shared_mem)
{
  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;

  // Use an additional group of threads to fill sorted_token_ids.
  // Since the current kernel will use sorted_token_ids afterward,
  // we fill sorted_token_ids within the same threadblock to make
  // synchronization easier.
  if (item.get_local_id(2) < fill_threads) {
    // Initialize sorted_token_ids with numel
    for (int it = item.get_local_id(2); it < max_num_tokens_padded;
         it += fill_threads) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    // Three __syncthreads() corresponding to the other threads
    item.barrier(sycl::access::fence_space::local_space);
    //item.barrier(sycl::access::fence_space::local_space);
    //item.barrier(sycl::access::fence_space::local_space);
    return;
  }

  const int tid = item.get_local_id(2) - fill_threads;
  const int stride = item.get_local_range(2) - fill_threads;

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

  item.barrier(sycl::access::fence_space::local_space);

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
          CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) * block_size;
    }
    total_tokens_post_pad[model_offset] =
        static_cast<int>(cumsum[num_experts]);
  }

  item.barrier(sycl::access::fence_space::local_space);

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
void _count_and_sort_expert_tokens(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ cumsum_buffer,
    int* __restrict__ expert_map, int numel, int num_experts,
    int max_num_tokens_padded, int* __restrict__ token_mask,
    int model_offset, int topk_num, const sycl::nd_item<3> &item)
{
  const int tid = item.get_group(1) * item.get_local_range(2) +
                  item.get_local_id(2);
  const int stride = item.get_local_range(2) * item.get_group_range(1);

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
      auto ao = sycl::atomic_ref<int, sycl::memory_order::relaxed, \
                                 sycl::memory_scope::device,\
                                 sycl::access::address_space::generic_space>(
               cumsum_buffer[(model_offset * (num_experts + 1)) + expert_id]);
      int rank_post_pad = ao.fetch_add(1);
      sorted_token_ids[max_num_tokens_padded * model_offset + rank_post_pad] = i;
    }
  }
}

template <typename scalar_t>
void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts,
    int padded_num_experts, int experts_per_warp, int block_size,
    int numel, int* __restrict__ cumsum, int max_num_tokens_padded,
    int topk_num, const sycl::nd_item<3> &item, int *shared_mem)
{
  _moe_align_block_size(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, padded_num_experts, experts_per_warp, block_size, numel,
      cumsum, max_num_tokens_padded, CEILDIV(max_num_tokens_padded, block_size),
      0, 0, topk_num, nullptr, item, shared_mem);
}



template <typename scalar_t>
void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ cumsum_buffer,
    int* __restrict__ expert_map, int numel, int num_experts,
    int max_num_tokens_padded, int topk_num, const sycl::nd_item<3> &item)
{
  _count_and_sort_expert_tokens(
      topk_ids, sorted_token_ids, cumsum_buffer, expert_map, numel, num_experts,
      max_num_tokens_padded, nullptr, 0, topk_num, item);
}

template <typename scalar_t, int fill_threads>
void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts, int block_size,
    int numel, int max_num_tokens_padded, int topk_num,
    const sycl::nd_item<3> &item, int *shared_mem)
{

  _moe_align_block_size_small_batch_expert<scalar_t, fill_threads>(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, expert_map,
      num_experts, block_size, numel, max_num_tokens_padded,
      CEILDIV(max_num_tokens_padded, block_size), 0, 0, topk_num, nullptr,
      item, shared_mem);
}

// taken from
// https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc
void moe_align_block_size(sycl::queue &q, int *topk_ids, int num_experts,
                          int block_size, int *sorted_token_ids,
                          int *experts_ids,
                          int *num_tokens_post_pad,
                          int *expert_map,
                          int topk_ids_size,
                          int topk,
                          int sorted_token_ids_size)
{
  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int WARP_SIZE = *r;

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
    const int shared_mem_size = ((threads + 1) * num_experts + (num_experts + 1));

    // threadIdx.x >= fill_threads: counting experts and aligning
    // threadIdx.x < fill_threads: filling sorted_token_ids
    constexpr int fill_threads = 256;

    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int, 1> shared_mem (sycl::range<1>(shared_mem_size), cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,fill_threads + threads),
                                         sycl::range<3>(1,1,fill_threads + threads)),
                       [=](sycl::nd_item<3> item) {
        moe_align_block_size_small_batch_expert_kernel<int, fill_threads>(
            topk_ids, sorted_token_ids, experts_ids, num_tokens_post_pad,
            expert_map, num_experts, block_size, topk_ids_size, sorted_token_ids_size,
            topk, item,
            shared_mem.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    }).wait();
  } else {
    int *cumsum_buff;
    cumsum_buff = sycl::malloc_device<int>((num_experts + 1), q);
    q.memset(cumsum_buff, 0, sizeof(int) * (num_experts + 1));

    int num_warps = CEILDIV(padded_num_experts, experts_per_warp);
    int shared_mem_size = num_warps * experts_per_warp;

    // launch two threadblocks
    // blockIdx.x == 0: counting experts and aligning
    // blockIdx.x == 1: filling sorted_token_ids
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int, 1> shared_mem(sycl::range<1>(shared_mem_size), cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,2*threads),
                                         sycl::range<3>(1,1,threads)),
                       [=](sycl::nd_item<3> item) {
        moe_align_block_size_kernel<int>(
            topk_ids, sorted_token_ids, experts_ids, num_tokens_post_pad,
            expert_map, num_experts, padded_num_experts, experts_per_warp,
            block_size, topk_ids_size, cumsum_buff, sorted_token_ids_size, topk, item,
            shared_mem.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });

    const int block_threads = std::min(256, (int)threads);
    const int num_blocks = (topk_ids_size + block_threads - 1) / block_threads;
    const int max_blocks = 65535;
    const int actual_blocks = std::min(num_blocks, max_blocks);
    sycl::range<3> lws (1, 1, block_threads);
    sycl::range<3> gws (1, actual_blocks, block_threads);

    q.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
      count_and_sort_expert_tokens_kernel<int>(
          topk_ids, sorted_token_ids, cumsum_buff, expert_map, topk_ids_size,
          num_experts, sorted_token_ids_size, topk, item);
    }).wait();

    sycl::free(cumsum_buff, q);
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

#ifdef USE_GPU
   sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
   sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

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

         topk_ids = sycl::malloc_device<int>(m * topk, q);
         q.memcpy(topk_ids, h_topk_ids, topk_ids_size * sizeof(int));

         int max_num_tokens_padded = topk_ids_size + num_experts * (block_size - 1);

         if (pad_sorted_ids)
             max_num_tokens_padded = CEILDIV(max_num_tokens_padded, block_size) * block_size;

         if (topk_ids_size < num_experts)
             max_num_tokens_padded = std::min(topk_ids_size * block_size, max_num_tokens_padded);

         #ifdef DEBUG
         printf("max_num_tokens_padded: %zu\n",  max_num_tokens_padded);
         #endif

         int *sorted_ids;
         sorted_ids = sycl::malloc_device<int>(max_num_tokens_padded, q);

         int *expert_ids;
         int max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size);
         expert_ids = sycl::malloc_device<int>(max_num_m_blocks, q);

         int *num_tokens_post_pad;
         num_tokens_post_pad = sycl::malloc_device<int>(1, q);

         //bool ignore_invalid_experts = false;

         auto start = std::chrono::steady_clock::now();

         for (int n = 0; n < repeat; n++) {
           moe_align_block_size(
                q,
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
         q.memcpy(&actual_num_tokens, num_tokens_post_pad, sizeof(int));

         int *actual_expert_ids = (int*) malloc(max_num_m_blocks * sizeof(int));
         q.memcpy(actual_expert_ids, expert_ids, max_num_m_blocks * sizeof(int));

         int *actual_sorted_ids;
         actual_sorted_ids = (int*) malloc (max_num_tokens_padded * sizeof(int));
         q.memcpy(actual_sorted_ids, sorted_ids, max_num_tokens_padded * sizeof(int));
         q.wait();

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

         sycl::free(topk_ids, q);
         sycl::free(sorted_ids, q);
         sycl::free(expert_ids, q);
         sycl::free(num_tokens_post_pad, q);

         free(h_topk_ids);
         free(actual_sorted_ids);
         free(actual_expert_ids);
       }
     }
   }
   if (!ok) return 1;
   return 0;
}
