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
#include <sycl/sycl.hpp>
#include "reference.h"

// The fused add-bias-QKV + transpose kernel is implemented with 
// software FP8 (E4M3) and bfloat16 conversions. The helpers
// below mirror the host reference in reference.h bit-for-bit so the GPU result
// is byte-exact with the reference. The OCP / E4M3FN encoding (fnuz == false)
// is used, matching the NVIDIA (CUDA) reference.
namespace dev {

inline float bits_to_f32(uint32_t b) { return sycl::bit_cast<float>(b); }
inline uint32_t f32_to_bits(float f) { return sycl::bit_cast<uint32_t>(f); }

// bfloat16 (truncated fp32, round-to-nearest-even)
inline float bf16_to_f32(uint16_t h) {
  return bits_to_f32(static_cast<uint32_t>(h) << 16);
}

inline uint16_t f32_to_bf16(float f) {
  uint32_t x = f32_to_bits(f);
  if (((x >> 23) & 0xFF) == 0xFF && (x & 0x7FFFFF) != 0)
    return static_cast<uint16_t>((x >> 16) | 0x0040u);
  uint32_t rounding_bias = ((x >> 16) & 1u) + 0x7FFFu;
  x += rounding_bias;
  return static_cast<uint16_t>(x >> 16);
}

inline float bf16_round(float f) { return bf16_to_f32(f32_to_bf16(f)); }

// FP8 E4M3 decode / encode
inline float fp8_e4m3_to_f32(uint8_t v, bool fnuz) {
  const int bias = fnuz ? 8 : 7;
  const uint32_t sign = (v >> 7) & 1u;
  const uint32_t exp  = (v >> 3) & 0xFu;
  const uint32_t mant = v & 0x7u;

  if (fnuz) {
    if (v == 0x80u) return sycl::nan(0u);
  } else {
    if (exp == 0xFu && mant == 0x7u) return sycl::nan(0u);
  }

  const float sgn = sign ? -1.0f : 1.0f;
  if (exp == 0u)
    return sgn * sycl::ldexp(static_cast<float>(mant), 1 - bias - 3);
  return sgn * sycl::ldexp(static_cast<float>(8u + mant),
                           static_cast<int>(exp) - bias - 3);
}

inline uint32_t round_to_even(float x) {
  float fl = sycl::floor(x);
  float diff = x - fl;
  uint32_t i = static_cast<uint32_t>(fl);
  if (diff < 0.5f) return i;
  if (diff > 0.5f) return i + 1u;
  return (i & 1u) ? i + 1u : i;
}

inline uint8_t f32_to_fp8_e4m3(float f, bool fnuz) {
  const int bias = fnuz ? 8 : 7;
  const uint32_t sign = (f32_to_bits(f) >> 31) & 1u;
  const float af = sycl::fabs(f);

  if (sycl::isnan(f) || sycl::isinf(f)) {
    if (fnuz) return 0x80u;
    return static_cast<uint8_t>((sign << 7) | 0x7Fu);
  }
  if (af == 0.0f) return fnuz ? 0x00u : static_cast<uint8_t>(sign << 7);

  const float max_normal = fnuz ? 240.0f : 448.0f;

  int e;
  float m = sycl::frexp(af, &e);
  int biased = (e - 1) + bias;
  float frac = m * 2.0f;

  if (biased <= 0) {
    float step = sycl::ldexp(1.0f, 1 - bias - 3);
    uint32_t q = round_to_even(af / step);
    if (q == 0u) return fnuz ? 0x00u : static_cast<uint8_t>(sign << 7);
    if (q >= 8u)
      return static_cast<uint8_t>((sign << 7) | (1u << 3) | ((q - 8u) & 7u));
    return static_cast<uint8_t>((sign << 7) | q);
  }

  uint32_t q = round_to_even((frac - 1.0f) * 8.0f);
  if (q == 8u) { q = 0u; biased += 1; }

  if (fnuz) {
    if (biased > 15 || af > max_normal)
      return static_cast<uint8_t>((sign << 7) | 0x7Fu);
  } else {
    if (biased > 15 || (biased == 15 && q >= 7u) || af > max_normal)
      return static_cast<uint8_t>((sign << 7) | 0x7Eu);
  }

  return static_cast<uint8_t>((sign << 7) | (static_cast<uint32_t>(biased) << 3) |
                              (q & 7u));
}

}  // namespace dev

// Fused add-bias-QKV + transpose kernel.
//
// Each work-item processes 4 consecutive FP8 elements (one group along the
// size_per_head dimension), matching the fp8x4 access of the CUDA/HIP kernels:
//   1. decode FP8 (E4M3) source to fp32
//   2. out = ((src * input_scale) + bias) * output_scale   (bfloat16 rounding)
//   3. encode the result back to FP8 (E4M3)
//   4. transpose [valid_word_num, 3, head, size] -> [valid_word_num, head, 3, size]
void addBiasQKV_kernel(
    const uint8_t*  src,
          uint8_t*  out,
    const uint16_t* bias,
    float           in_s,
    float           out_s,
    int             head_num,
    int             size_per_head,
    int             hidden_unit,
    int             total_groups,
    bool            fnuz,
    sycl::nd_item<1> &item)
{
  int gid = item.get_global_id(0);
  if (gid >= total_groups) return;

  const int size_div_4   = size_per_head / 4;
  const int hidden_div_4 = hidden_unit / 4;

  int x = gid % size_div_4;
  int t = gid / size_div_4;
  int y = t % head_num;                 // head
  t /= head_num;
  int z = t % 3;                        // q / k / v
  int w = t / 3;                        // word

  const int src_id   = z * hidden_div_4 + y * size_div_4 + x;
  const int src_base = w * 3 * hidden_unit + src_id * 4;
  const int bias_base = z * hidden_unit + y * size_per_head + x * 4;
  const int out_base = w * 3 * hidden_unit +
                       y * (3 * size_per_head) + z * size_per_head + x * 4;

  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    float v = dev::bf16_round(dev::fp8_e4m3_to_f32(src[src_base + k], fnuz));
    v = dev::bf16_round(v * in_s);
    v = dev::bf16_round(v + dev::bf16_to_f32(bias[bias_base + k]));
    v = dev::bf16_round(v * out_s);
    out[out_base + k] = dev::f32_to_fp8_e4m3(v, fnuz);
  }
}

void addBias(int batch_size, int seq_len, int hidden_units, int head_num, int repeat) {

  int m = batch_size * seq_len;
  int size_per_head = hidden_units / head_num;
  int bias_size = 3 * hidden_units;
  const int total = m * 3 * hidden_units;

  // Intel/SYCL software path uses the OCP / E4M3FN encoding (matches CUDA).
  const bool fnuz = false;

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
  const float in_s  = ref::bf16_to_f32(ref::f32_to_bf16(input_scale));
  const float out_s = ref::bf16_to_f32(ref::f32_to_bf16(output_scale));

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  uint8_t*  d_src  = sycl::malloc_device<uint8_t>(total, q);
  uint8_t*  d_out  = sycl::malloc_device<uint8_t>(total, q);
  uint16_t* d_bias = sycl::malloc_device<uint16_t>(bias_size, q);

  q.memcpy(d_src, h_src.data(), sizeof(uint8_t) * total);
  q.memcpy(d_bias, h_bias.data(), sizeof(uint16_t) * bias_size);
  q.wait();

  const int total_groups = total / 4;
  const int block_size = 256;
  const int grid = (total_groups + block_size - 1) / block_size;
  sycl::nd_range<1> ndr(sycl::range<1>((size_t)grid * block_size),
                        sycl::range<1>(block_size));

  auto launch = [&]() {
    q.parallel_for(ndr, [=](sycl::nd_item<1> item) {
      addBiasQKV_kernel(d_src, d_out, d_bias, in_s, out_s,
                        head_num, size_per_head, hidden_units,
                        total_groups, fnuz, item);
    });
  };

  // warmup
  for (int i = 0; i < repeat; i++) launch();
  q.wait();

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) launch();
  q.wait();
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

  // Verify the GPU result against the host reference.
  q.memcpy(h_out.data(), d_out, sizeof(uint8_t) * total).wait();

  addBiasQKV_reference(h_ref.data(), h_src.data(), h_bias.data(),
                       input_scale, output_scale,
                       m, head_num, size_per_head, hidden_units, fnuz);

  int mismatches = 0;
  for (int i = 0; i < total; i++)
    if (h_out[i] != h_ref[i]) mismatches++;

  printf("%s\n", mismatches == 0 ? "PASS" : "FAIL");

  sycl::free(d_src, q);
  sycl::free(d_out, q);
  sycl::free(d_bias, q);
}

int main(int argc, char **argv) {
  int batch_size = 8;
  int seq_len = 1024;
  int hidden_units = 768;
  int head_num = 12;
  int repeat = 1000;
  printf("Using FP8 E4M3 format: OCP\n");
  addBias(batch_size, seq_len, hidden_units, head_num, repeat);
  return 0;
}
