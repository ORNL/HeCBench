#include <cassert>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include "reference.h"

#define C10_WARP_SIZE 32
typedef __hip_bfloat16 BFloat16;
typedef __half Half;

template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void copy_vector(Datatype *dst, const Datatype *src);

template <>
__device__ __inline__ void copy_vector<BFloat16, 1>(BFloat16 *dst,
                                                    const BFloat16 *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<BFloat16, 4>(BFloat16 *dst,
                                                    const BFloat16 *src) {
  *((float2 *)dst) = *((float2 *)src);
}

template <>
__device__ __inline__ void copy_vector<Half, 1>(Half *dst, const Half *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<Half, 4>(Half *dst, const Half *src) {
  *((float2 *)dst) = *((float2 *)src);
}

template <>
__device__ __inline__ void copy_vector<uint8_t, 1>(uint8_t *dst,
                                                   const uint8_t *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<uint8_t, 4>(uint8_t *dst,
                                                   const uint8_t *src) {
  *((uchar4 *)dst) = *((uchar4 *)src);
}

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value)
    ++log2_value;
  return log2_value;
}

template <typename T> struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T> struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
__device__ __forceinline__ T
WARP_SHFL_XOR_NATIVE(T value, int laneMask, int width = warpSize) {
  return __shfl_xor(value, laneMask, width);
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE,
          template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t *sum) {
  ReduceOp<acc_t> r;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      acc_t b = WARP_SHFL_XOR_NATIVE(sum[i], offset, WARP_SIZE);
      sum[i] = r(sum[i], b);
    }
  }
}

/*
 * Extended softmax (from native aten pytorch) with following additional
 * features 1) input scaling 2) Explicit masking
 */

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
__global__ void scaled_masked_softmax_warp_forward(
    output_t *dst, const input_t *src, const uint8_t *mask, const acc_t scale,
    int micro_batch_size, int element_count, int pad_batches) {
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE =
      (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
  constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

  // blockDim/threadIdx = (WARP_SIZE, WARPS_PER_BLOCK, 1)
  // gridDim/blockIdx = (seq_len, attn_heads, batches)
  int first_batch = (blockDim.y * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) + threadIdx.y) * WARP_BATCH;

  int pad_first_batch;
  if (pad_batches != 1) { // bert style
    pad_first_batch = (blockDim.y * (blockIdx.x + gridDim.x * blockIdx.z) + threadIdx.y) * WARP_BATCH;
  } else { // gpt2 style
    pad_first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
  }

  // micro_batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = micro_batch_size - first_batch;
  if (local_batches > WARP_BATCH)
    local_batches = WARP_BATCH;

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  src += first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;
  dst += first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;
  mask += pad_first_batch * element_count + ELEMENTS_PER_LDG_STG * local_idx;

  // load data from global memory
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  input_t temp_data[ELEMENTS_PER_LDG_STG];
  uint8_t temp_mask[ELEMENTS_PER_LDG_STG];
#pragma unroll
  for (int i = 0; i < local_batches; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {

        int itr_idx = i * element_count + it * WARP_SIZE;
        copy_vector<input_t, ELEMENTS_PER_LDG_STG>(temp_data, src + itr_idx);
        copy_vector<uint8_t, ELEMENTS_PER_LDG_STG>(temp_mask, mask + itr_idx);

#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          if (temp_mask[element] != 1) {
            elements[i][it + element] = (acc_t)temp_data[element] * scale;
          } else {
            elements[i][it + element] = (acc_t)-10000.0;
          }
        }
      } else {
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          elements[i][it + element] = -std::numeric_limits<acc_t>::infinity();
        }
      }
    }
  }

  // compute max_value
  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

  // compute scale value to account for full mask
  acc_t mask_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    mask_value[i] = (max_value[i] == (acc_t)-10000.0) ? (acc_t)0.0 : (acc_t)1.0;
  }


  acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = std::exp((elements[i][it] - max_value[i]));
      sum[i] += elements[i][it];
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

  // store result
  output_t out[ELEMENTS_PER_LDG_STG];
#pragma unroll
  for (int i = 0; i < local_batches; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; it += ELEMENTS_PER_LDG_STG) {
      int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
#pragma unroll
        for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
          if (mask_value[i])
            out[element] = elements[i][it + element] / sum[i];
          else
            out[element] = (output_t)0;
        }
        copy_vector<output_t, ELEMENTS_PER_LDG_STG>(
            dst + i * element_count + it * WARP_SIZE, out);
      } else {
        break;
      }
    }
  }
}

#define LAUNCH(n)                                                              \
  scaled_masked_softmax_warp_forward<input_t, output_t, acc_t, n>              \
      <<<blocks, threads>>>(dst, src, mask, scale, batch_count, key_seq_len,   \
                            pad_batches);

template <typename input_t, typename output_t, typename acc_t>
void scaled_masked_softmax_forward(output_t *dst, const input_t *src,
                                   const uint8_t *mask, const acc_t scale,
                                   int query_seq_len, int key_seq_len,
                                   int batches, int attn_heads,
                                   int pad_batches) {
  assert(key_seq_len >= 0 && key_seq_len <= 1024);
  if (key_seq_len == 0) {
    return;
  } else {
    int log2_elements = log2_ceil(key_seq_len);
    const int next_power_of_two = 1 << log2_elements;
    int batch_count = batches * attn_heads * query_seq_len;

    int warp_size =
        (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    assert(query_seq_len % batches_per_block == 0);

    dim3 blocks(query_seq_len / batches_per_block, attn_heads, batches);
    dim3 threads(warp_size, warps_per_block, 1);
    switch (log2_elements) {
    case 0: LAUNCH(0); break;
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    case 3: LAUNCH(3); break;
    case 4: LAUNCH(4); break;
    case 5: LAUNCH(5); break;
    case 6: LAUNCH(6); break;
    case 7: LAUNCH(7); break;
    case 8: LAUNCH(8); break;
    case 9: LAUNCH(9); break;
    case 10: LAUNCH(10); break;
    default: break;
    }
  }
}

template <typename scalar_t>
void fused_softmax(int batches, int attn_heads, int query_seq_len,
                   int key_seq_len, int pad_batches, int repeat) {
  uint64_t num_data_elems =
      (uint64_t)batches * attn_heads * query_seq_len * key_seq_len;
  float scale_factor = 1.f / sqrtf(key_seq_len);
  uint8_t *mask = (uint8_t*) malloc (sizeof(uint8_t) * num_data_elems);

  srand(123);
  for (size_t i = 0; i < (uint64_t)batches * attn_heads * query_seq_len; i++) {
    uint8_t* mask_row = mask + i * key_seq_len;
    int len = rand() % (key_seq_len / 2);
    memset(mask_row, 1, sizeof(uint8_t) * len); // mask
    memset(mask_row + len, 0, sizeof(uint8_t) * (key_seq_len - len));
  }

  scalar_t *input = make_random<scalar_t>(num_data_elems);

  const int* outliers = make_random_int((uint64_t)batches * attn_heads * query_seq_len, key_seq_len);
  for(int k = 0; k < query_seq_len; ++k) {
    for(int j = 0; j < batches * attn_heads; ++j) {
      input[j * key_seq_len + outliers[j* query_seq_len + k]] *= scalar_t(20);
    }
  }

  scalar_t *output = (scalar_t *)aligned_alloc(1024, sizeof(scalar_t) * num_data_elems);
  scalar_t *output_ref = (scalar_t *)aligned_alloc(1024, sizeof(scalar_t) * num_data_elems);

  scalar_t *d_input, *d_output;
  uint8_t *d_mask;
  hipMalloc((void **)&d_input, sizeof(scalar_t) * num_data_elems);
  hipMalloc((void **)&d_output, sizeof(scalar_t) * num_data_elems);
  hipMalloc((void **)&d_mask, sizeof(uint8_t) * num_data_elems);
  hipMemcpy(d_mask, mask, sizeof(uint8_t) * num_data_elems, hipMemcpyHostToDevice);
  hipMemcpy(d_input, input, sizeof(scalar_t) * num_data_elems, hipMemcpyHostToDevice);

  scaled_masked_softmax_forward<scalar_t, scalar_t, float>(
      d_output, d_input, d_mask, scale_factor, query_seq_len, key_seq_len,
      batches, attn_heads, pad_batches);

  hipMemcpy(output, d_output, sizeof(scalar_t) * num_data_elems, hipMemcpyDeviceToHost);

  reference<scalar_t, scalar_t, float>(output_ref, input, mask, scale_factor,
                                       pad_batches, batches, attn_heads, query_seq_len, key_seq_len);
  bool ok = true;
  for (size_t i = 0; i < num_data_elems; i++) {
    if (fabsf((float)output[i] - (float)output_ref[i]) > 1e-3f) {
      printf("Mismatch at index %zu: %f %f\n", i, (float)output[i], (float)output_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    scaled_masked_softmax_forward<scalar_t, scalar_t, float>(
        d_output, d_input, d_mask, scale_factor, query_seq_len, key_seq_len,
        batches, attn_heads, pad_batches);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  free(output);
  free(output_ref);
  free(input);
  free(mask);
  hipFree(d_input);
  hipFree(d_output);
  hipFree(d_mask);
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf("Usage: %s <batch> <head> <query length> <key length> <repeat>\n",
           argv[0]);
    return 1;
  }

  int batches = atoi(argv[1]);
  int attn_heads = atoi(argv[2]);
  int query_seq_len = atoi(argv[3]);
  int key_seq_len = atoi(argv[4]);
  int repeat = atoi(argv[5]);

  // bert mask
  fused_softmax<Half>(batches, attn_heads, query_seq_len, key_seq_len, 0, repeat);
  // gpt2 mask
  fused_softmax<Half>(batches, attn_heads, query_seq_len, key_seq_len, 1, repeat);
  fused_softmax<BFloat16>(batches, attn_heads, query_seq_len, key_seq_len, 0, repeat);
  fused_softmax<BFloat16>(batches, attn_heads, query_seq_len, key_seq_len, 1, repeat);

  if (!ok) return 1;
  return 0;
}
