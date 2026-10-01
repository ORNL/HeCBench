// Off-by-one tolerance verification for int8 quantization results.
// Host (reference) and device may differ by a small amount at quantization
// midpoints due to fp32 rounding/division precision (device fp32 ops are only
// required to be accurate to within a few ULPs). This mirrors vLLM's upstream
// int8-quant tests, which verify with torch.testing.assert_close(atol=1/2,
// rtol=0) rather than bit-exact equality. See intel/llvm issue #16636.
// Returns 0 if all elements are within atol, non-zero otherwise.
static inline int compare_int8_tol(const int8_t* a, const int8_t* b,
                                   size_t n, int atol) {
  for (size_t i = 0; i < n; i++) {
    int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
    if (d < 0) d = -d;
    if (d > atol) return 1;
  }
  return 0;
}

template <typename scalar_t, typename scale_type>
void static_scaled_int8_quant_reference(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type const scale,
    const int64_t num_tokens, 
    const int hidden_size)
{
  for (int64_t token_idx = 0; token_idx < num_tokens; token_idx++) {
    for (int i = 0; i < hidden_size; i++) {
      out[token_idx * hidden_size + i] = float_to_int8_rn(static_cast<float>(input[token_idx * hidden_size + i]) / scale);
    }
  }
}

template <typename scalar_t, typename scale_type, typename azp_type>
void static_scaled_int8_azp_quant_reference(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type const scale, azp_type const azp,
    const int64_t num_tokens, 
    const int hidden_size)
{
  for (int64_t token_idx = 0; token_idx < num_tokens; token_idx++) {
    for (int i = 0; i < hidden_size; i++) {
      auto const val = static_cast<float>(input[token_idx * hidden_size + i]);
      auto const quant_val = int32_to_int8(float_to_int32_rn(val / scale) + azp);
      out[token_idx * hidden_size + i] = quant_val; 
    }
  }
}

template <typename scalar_t, typename scale_type>
void dynamic_scaled_int8_quant_reference(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale,
    const int64_t num_tokens, 
    const int hidden_size)
{
  for (int64_t token_idx = 0; token_idx < num_tokens; token_idx++) {

    float absmax_val = 0.0f;
    float const zero = 0.0f;

    for (int i = 0; i < hidden_size; i++) {
      float val = static_cast<float>(input[token_idx * hidden_size + i]);
      val = val > zero ? val : -val;
      absmax_val = val > absmax_val ? val : absmax_val;
    }
    scale[token_idx] = absmax_val / 127.0f;

    float const tmp_scale = 127.0f / absmax_val;
    for (int i = 0; i < hidden_size; i++) {
      out[token_idx * hidden_size + i] = float_to_int8_rn(static_cast<float>(
        input[token_idx * hidden_size + i]) * tmp_scale);
    }
  }
}

template <typename scalar_t, typename scale_type, typename azp_type>
void dynamic_scaled_int8_azp_quant_reference(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, azp_type* azp, 
    const int64_t num_tokens, 
    const int hidden_size)
{
  for (int64_t token_idx = 0; token_idx < num_tokens; token_idx++) {

    // Scan for the min and max value for this token
    float max_val = std::numeric_limits<float>::min();
    float min_val = std::numeric_limits<float>::max();
    for (int i = 0; i < hidden_size; i++) {
      auto val = static_cast<float>(input[token_idx * hidden_size + i]);
      max_val = std::max(max_val, val);
      min_val = std::min(min_val, val);
    }

    float const scale_val = (max_val - min_val) / 255.0f;
    // Use rounding to even (same as torch.round)
    auto const azp_float = std::nearbyint(-128.0f - min_val / scale_val);
    auto const azp_val = static_cast<azp_type>(azp_float);

    // Store the scale and azp into shared and global
    scale[token_idx] = scale_val;
    azp[token_idx] = azp_val;

    // Quantize the values
    for (int i = 0; i < hidden_size; i++) {
      auto val = static_cast<float>(input[token_idx * hidden_size + i]);
      auto const quant_val =
          int32_to_int8(float_to_int32_rn(val / scale_val) + azp_val);
      out[token_idx * hidden_size + i] = quant_val;
    }
  }
}
