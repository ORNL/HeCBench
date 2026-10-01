/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include "reference.h"

// FP8 E4M3 encoding differs by GPU architecture:
//   gfx940/gfx942 (MI300)  -> FNUZ  (__hip_fp8*_e4m3_fnuz)
//   gfx950        (MI350)  -> OCP   (__hip_fp8*_e4m3)
// The __gfx__ preprocessor macros are only defined during device compilation,
// so the arch-specific FP8 type used inside the kernel is selected here at
// compile time. Both formats share the same 1-byte storage layout, so the
// host-side storage tag (see addBias) stays fixed and only the in-kernel
// encode/decode differs.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
using fp8_e4m3_t   = __hip_fp8_e4m3;
using fp8x4_e4m3_t = __hip_fp8x4_e4m3;
#else
using fp8_e4m3_t   = __hip_fp8_e4m3_fnuz;
using fp8x4_e4m3_t = __hip_fp8x4_e4m3_fnuz;
#endif

// Host-side detection of the FP8 format the current device uses (for logging).
static bool fp8IsOcp() {
  int device = 0;
  hipGetDevice(&device);
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, device);
  return std::string(prop.gcnArchName).find("gfx95") != std::string::npos;
}

template<typename T1, typename T2>
struct FP8TrtAddQKVBiasParam {
    T1*          qkv_tgt;
    const T1*    qkv_src;
    const T2*    qkv_bias;
    const float* input_scale;
    const float* output_scale;
    const int    valid_word_num;
    const int    head_num;
    const int    size_per_head;
    const int    hidden_unit;
};

inline __device__ __hip_bfloat162 hmul2(__hip_bfloat162 x, __hip_bfloat162 y) {
  return __hmul2(x, y);
}
inline __device__ __hip_bfloat162 hadd2(__hip_bfloat162 x, __hip_bfloat162 y) {
  return __hadd2(x, y);
}

// not necessarily defined in HIP
inline __device__ __hip_bfloat162 __float2bfloat162_rn(const float a) {
  return __hip_bfloat162{__float2bfloat16(a), __float2bfloat16(a)};
}

inline __device__ __hip_bfloat162 float_to_bfloat2(float val) {
  return __float2bfloat162_rn(val);
}

inline __device__ void fp8x4_e4m3_to_bfloat2(__hip_bfloat162* out1, __hip_bfloat162* out2, const fp8x4_e4m3_t* in)
{
  const char4 tmp_val = reinterpret_cast<const char4*>(in)[0];
  *out1 = __hip_bfloat162((float)reinterpret_cast<const fp8_e4m3_t*>(&tmp_val.x)[0],
                          (float)reinterpret_cast<const fp8_e4m3_t*>(&tmp_val.y)[0]);
  *out2 = __hip_bfloat162((float)reinterpret_cast<const fp8_e4m3_t*>(&tmp_val.z)[0],
                          (float)reinterpret_cast<const fp8_e4m3_t*>(&tmp_val.w)[0]);
}


__global__ void FP8TrtAddQKVBiasKernel(FP8TrtAddQKVBiasParam<__hip_fp8_e4m3_fnuz, __hip_bfloat16> param)
{
    // Add bias ([3, head, size]), and then transpose from
    // [valid_word_num, 3, head, size] -> [valid_word_num, head, 3, size]

    using T1_4 = fp8x4_e4m3_t;
    using T2_2 = __hip_bfloat162;

    const T1_4* qkv_src_ptr = (T1_4*)(param.qkv_src + blockIdx.x * 3 * param.hidden_unit);
    const T2_2* bias_ptr    = (T2_2*)param.qkv_bias;
    T1_4*       qkv_tgt_ptr = (T1_4*)(param.qkv_tgt + blockIdx.x * 3 * param.hidden_unit);

    const int size_div_4   = param.size_per_head / 4;
    const int hidden_div_4 = param.hidden_unit / 4;
    const int src_id       = threadIdx.z * hidden_div_4 + threadIdx.y * size_div_4 + threadIdx.x;

    T2_2 val1, val2;
    fp8x4_e4m3_to_bfloat2(&val1, &val2, &qkv_src_ptr[src_id]);
    T2_2      input_scale_2  = float_to_bfloat2(__ldg(param.input_scale)); 
    T2_2      output_scale_2 = float_to_bfloat2(__ldg(param.output_scale));
    const int bias_id_0      = src_id * 2;
    val1                     = hmul2(hadd2(hmul2(val1, input_scale_2), bias_ptr[bias_id_0]), output_scale_2);
    val2                     = hmul2(hadd2(hmul2(val2, input_scale_2), bias_ptr[bias_id_0 + 1]), output_scale_2);

    // NOTE: HIP's fp8x4(low, high) constructor places the first argument in the
    // high bytes and the second in the low bytes -- the reverse of NVIDIA's
    // __nv_fp8x4_e4m3(x, y). Swap the halves so the 4 packed elements keep the
    // same in-memory order as the CUDA kernel.
    qkv_tgt_ptr[(threadIdx.y * 3 * size_div_4 + threadIdx.z * size_div_4) + threadIdx.x] = fp8x4_e4m3_t(val2, val1);
}

template<typename T1, typename T2>
void addBias(int batch_size, int seq_len, int hidden_units, int head_num, int repeat) {

  int m = batch_size * seq_len;

  T1 *qkv_buf, *q_buf;
  hipMalloc(&qkv_buf, sizeof(T1) * m * 3 * hidden_units);
  hipMalloc(&q_buf, sizeof(T1) * m * 3 * hidden_units);

  int size_per_head = hidden_units / head_num;
  int bias_size = 3 * hidden_units;

  T2 *bias;
  float *oscale, *oscale_inv;

  hipMalloc(&bias, sizeof(T2) * bias_size);
  hipMalloc(&oscale, sizeof(float));
  hipMalloc(&oscale_inv, sizeof(float));

  // Host-side inputs (kept in device storage formats: FP8 bytes / bf16 words).
  const bool fnuz = !fp8IsOcp();
  const int total = m * 3 * hidden_units;

  std::vector<uint8_t>  h_src(total);
  std::vector<uint16_t> h_bias(bias_size);
  std::vector<uint8_t>  h_out(total);
  std::vector<uint8_t>  h_ref(total);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (int i = 0; i < total; i++)
    h_src[i] = ref::f32_to_fp8_e4m3(dist(rng), fnuz);
  for (int i = 0; i < bias_size; i++)
    h_bias[i] = ref::f32_to_bf16(dist(rng));

  const float input_scale  = 0.5f;
  const float output_scale = 1.25f;

  hipMemcpy(qkv_buf, h_src.data(), sizeof(T1) * total, hipMemcpyHostToDevice);
  hipMemcpy(bias, h_bias.data(), sizeof(T2) * bias_size, hipMemcpyHostToDevice);
  hipMemcpy(oscale, &input_scale, sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(oscale_inv, &output_scale, sizeof(float), hipMemcpyHostToDevice);

  FP8TrtAddQKVBiasParam<T1, T2> param{q_buf,
                                      qkv_buf,
                                      bias,   //att_query_weight_bias,
                                      oscale, //att_query_weight_output_scale,
                                      oscale_inv, //att_query_weight_output_scale_inv,
                                      m,
                                      head_num,
                                      size_per_head,
                                      head_num * size_per_head};

  dim3 grid(param.valid_word_num);
  dim3 block(param.size_per_head / 4, param.head_num, 3);

  // warmup
  for (int i = 0; i < repeat; i++)
    FP8TrtAddQKVBiasKernel<<<grid, block>>>(param);

  hipDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++)
    FP8TrtAddQKVBiasKernel<<<grid, block>>>(param);

  hipDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

  // Verify the GPU result against the host reference.
  hipMemcpy(h_out.data(), q_buf, sizeof(T1) * total, hipMemcpyDeviceToHost);

  addBiasQKV_reference(h_ref.data(), h_src.data(), h_bias.data(),
                       input_scale, output_scale,
                       m, head_num, size_per_head, hidden_units, fnuz);

  int mismatches = 0;
  for (int i = 0; i < total; i++)
    if (h_out[i] != h_ref[i]) mismatches++;

  printf("%s\n", mismatches == 0 ? "PASS" : "FAIL");

  hipFree(qkv_buf);
  hipFree(q_buf);
  hipFree(bias);
  hipFree(oscale);
  hipFree(oscale_inv);
}

int main(int argc, char **argv) {
  int batch_size = 8;
  int seq_len = 1024;
  int hidden_units = 768;
  int head_num = 12;
  int repeat = 1000;
  printf("Using FP8 E4M3 format: %s\n", fp8IsOcp() ? "OCP" : "FNUZ");
  addBias<__hip_fp8_e4m3_fnuz, __hip_bfloat16>(batch_size, seq_len, hidden_units, head_num, repeat);
  return 0;
}
