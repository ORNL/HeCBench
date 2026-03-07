#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/util_type.cuh>
#include <cub/cub.cuh>
#include "utils.h"
#include "reference.h"
#include <cstdlib>

template <typename scalar_t, typename scale_type>
__global__ void static_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type const scale, const int hidden_size) {
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.x;

  // Performed using 64-bit math to avoid integer overflow.
  out += token_idx * hidden_size;
  input += token_idx * hidden_size;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[i] = float_to_int8_rn(static_cast<float>(input[i]) / scale);
  }
}

template <typename scalar_t, typename scale_type, typename azp_type>
__global__ void static_scaled_int8_azp_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type const scale, azp_type const azp,
    const int hidden_size) {
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.x;

  // Performed using 64-bit math to avoid integer overflow.
  out += token_idx * hidden_size;
  input += token_idx * hidden_size;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    auto const val = static_cast<float>(input[i]);
    auto const quant_val = int32_to_int8(float_to_int32_rn(val / scale) + azp);
    out[i] = quant_val;
  }
}

template <typename scalar_t, typename scale_type>
__global__ void dynamic_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size) {
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;

  // Performed using 64-bit math to avoid integer overflow.
  out += token_idx * hidden_size;
  input += token_idx * hidden_size;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(input[i]);
    val = val > zero ? val : -val;
    absmax_val = val > absmax_val ? val : absmax_val;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val / 127.0f;
  }
  __syncthreads();

  float const tmp_scale = 127.0f / block_absmax_val;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[i] = float_to_int8_rn(static_cast<float>(input[i]) * tmp_scale);
  }
}

template <typename scalar_t, typename scale_type, typename azp_type>
__global__ void dynamic_scaled_int8_azp_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, azp_type* azp, const int hidden_size) {
  int64_t const token_idx = blockIdx.x;

  // Performed using 64-bit math to avoid integer overflow.
  out += token_idx * hidden_size;
  input += token_idx * hidden_size;

  // Scan for the min and max value for this token
  float max_val = std::numeric_limits<float>::min();
  float min_val = std::numeric_limits<float>::max();
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    auto val = static_cast<float>(input[i]);
    max_val = std::max(max_val, val);
    min_val = std::min(min_val, val);
  }

  // Reduce the max and min values across the block
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  max_val = BlockReduce(reduceStorage).Reduce(max_val, Max{}, blockDim.x);
  __syncthreads();  // Make sure min doesn't mess with max shared memory
  min_val = BlockReduce(reduceStorage).Reduce(min_val, Min{}, blockDim.x);

  __shared__ scale_type scale_sh;
  __shared__ azp_type azp_sh;

  // Compute the scale and zero point and store them, only on the first thread
  if (threadIdx.x == 0) {
    float const scale_val = (max_val - min_val) / 255.0f;
    // Use rounding to even (same as torch.round)
    auto const azp_float = std::nearbyint(-128.0f - min_val / scale_val);
    auto const azp_val = static_cast<azp_type>(azp_float);

    // Store the scale and azp into shared and global
    scale[token_idx] = scale_sh = scale_val;
    azp[token_idx] = azp_sh = azp_val;
  }

  // Wait for the scale and azp to be computed
  __syncthreads();

  float const scale_val = scale_sh;
  azp_type const azp_val = azp_sh;

  // Quantize the values
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    auto const val = static_cast<float>(input[i]);
    auto const quant_val =
        int32_to_int8(float_to_int32_rn(val / scale_val) + azp_val);
    out[i] = quant_val;
  }
}


template <typename scalar_t>
void static_scaled_int8_quant(int8_t *output,
                              const scalar_t *input,
                              int const hidden_size,
                              int const num_tokens,
                              float scale,
                              int32_t azp)
{
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));

  if (!azp)
    static_scaled_int8_quant_kernel<scalar_t, float>
              <<<grid, block>>>(input, output, scale, hidden_size);
  else
    static_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t>
              <<<grid, block>>>(input, output, scale, azp, hidden_size);
}

template <typename scalar_t>
void dynamic_scaled_int8_quant(int8_t *output,
                               const scalar_t *input,
                               int const hidden_size,
                               int const num_tokens,
                               float *scales,
                               int32_t *azp)
{
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));
  if (azp == nullptr) {
    dynamic_scaled_int8_quant_kernel<scalar_t, float>
              <<<grid, block>>>(input, output, scales, hidden_size);
  } else {
    dynamic_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t>
              <<<grid, block>>>(input, output, scales, azp, hidden_size);
  }
}

template <typename scalar_t>
void quant(int num_tokens, int hidden_size, int repeat) {
  
  scalar_t *d_input, *h_input;
  float *d_scale, *h_scale;
  int32_t *d_azp = nullptr, *h_azp;
  int8_t *d_output, *h_output, *h_output_r;


  size_t input_size_bytes = hidden_size * num_tokens * sizeof(scalar_t);
  size_t output_size_bytes = hidden_size * num_tokens * sizeof(int8_t);
  size_t scale_size_bytes = num_tokens * sizeof(float);
  size_t azp_size_bytes = num_tokens * sizeof(int32_t);

  srand(123);
  h_input = (scalar_t*) malloc (input_size_bytes);
  for (int i = 0; i < hidden_size * num_tokens; i++) {
    h_input[i] = rand() % 1000 - 300;
  }
  
  h_scale = (float*) malloc (scale_size_bytes);
  h_azp = (int32_t*) malloc (azp_size_bytes);
  h_output = (int8_t*) malloc (output_size_bytes);
  h_output_r = (int8_t*) malloc (output_size_bytes);

  cudaMalloc(&d_input, input_size_bytes);
  cudaMalloc(&d_output, output_size_bytes);
  cudaMalloc(&d_scale, scale_size_bytes);

  cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);

  float scale = 0.1;
  int32_t error = 0;

  // static_scaled_int8_quant
  int32_t azp = 0;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) 
    static_scaled_int8_quant(d_output, d_input, hidden_size, num_tokens, scale, azp);
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of static_scaled_int8_quant kernel: %f (us)\n",
          (time * 1e-3f) / repeat);
  cudaMemcpy(h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);
  static_scaled_int8_quant_reference(h_input, h_output_r, scale, num_tokens, hidden_size);
  error = memcmp(h_output, h_output_r, output_size_bytes);

  // static_scaled_int8_quant_azp
  azp = 54;
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) 
    static_scaled_int8_quant(d_output, d_input, hidden_size, num_tokens, scale, azp);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of static_scaled_int8_quant_azp kernel: %f (us)\n",
          (time * 1e-3f) / repeat);
  cudaMemcpy(h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);
  static_scaled_int8_azp_quant_reference(h_input, h_output_r, scale, azp, num_tokens, hidden_size);
  error += memcmp(h_output, h_output_r, output_size_bytes);

  // dynamic_scaled_int8_quant
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) 
    dynamic_scaled_int8_quant(d_output, d_input, hidden_size, num_tokens, d_scale, d_azp);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of dynamic_scaled_int8_quant kernel: %f (us)\n",
          (time * 1e-3f) / repeat);
  cudaMemcpy(h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);
  dynamic_scaled_int8_quant_reference(h_input, h_output_r, h_scale, num_tokens, hidden_size);
  error += memcmp(h_output, h_output_r, output_size_bytes);

  // dynamic_scaled_int8_quant_azp
  cudaMalloc(&d_azp, azp_size_bytes);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) 
    dynamic_scaled_int8_quant(d_output, d_input, hidden_size, num_tokens, d_scale, d_azp);
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of dynamic_scaled_int8_quant_azp kernel: %f (us)\n",
          (time * 1e-3f) / repeat);
  cudaMemcpy(h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);
  dynamic_scaled_int8_azp_quant_reference(h_input, h_output_r, h_scale, h_azp, num_tokens, hidden_size);
  error += memcmp(h_output, h_output_r, output_size_bytes);

  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_scale);
  cudaFree(d_azp);
  free(h_input);
  free(h_output);
  free(h_output_r);
  free(h_scale);
  free(h_azp);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of tokens> <hidden size> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_tokens = atoi(argv[1]);
  const int hidden_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  printf("Input type is FP16\n");
  quant<__half>(num_tokens, hidden_size, repeat);

  printf("Input type is BF16\n");
  quant<__nv_bfloat16>(num_tokens, hidden_size, repeat);

  printf("Input type is FP32\n");
  quant<float>(num_tokens, hidden_size, repeat);
}
