#include <cmath>
#include <cstdint>
#include <cstdio>

template <typename scalar_t, typename accscalar_t, typename index_t>
void reference(
    scalar_t* output,
    scalar_t* total_weight,
    const scalar_t* input,
    const index_t*  target,
    const scalar_t* weights,
    bool size_average,
    int64_t nframe,
    int64_t kdim,
    int64_t ignore_index)
{
  accscalar_t output_acc = 0;
  accscalar_t total_weight_acc = 0;
  for (int i = 0; i < nframe; i++) {
    index_t t = target[i];
    if (t != ignore_index) {
      accscalar_t cur_weight =
          weights != nullptr ? static_cast<accscalar_t>(weights[t])
                             : static_cast<accscalar_t>(1);
      output_acc -= static_cast<accscalar_t>(input[i * kdim + t]) * cur_weight;
      total_weight_acc += cur_weight;
    }
  }
  *total_weight = static_cast<scalar_t>(total_weight_acc);
  if (size_average) {
    *output = static_cast<scalar_t>(output_acc / total_weight_acc);
  } else {
    *output = static_cast<scalar_t>(output_acc);
  }
}

// FP32 tolerance is relative. A blocked reduction and a sequential sum of the
// same values can disagree by several units in the averaged loss when the
// weight sum nearly cancels, while the relative error stays below one percent.
// The absolute allowance of 1 is only for FP16 accumulation.
template <typename scalar_t>
inline float nll_tolerance(float reference_value)
{
  if (sizeof(scalar_t) == sizeof(float))
    return 1e-4f + 1e-2f * std::fabs(reference_value);
  return 1.f;
}

template <typename scalar_t>
inline bool nll_results_match(scalar_t output,
                              scalar_t total_weight,
                              scalar_t ref_output,
                              scalar_t ref_total_weight)
{
  const float ho = static_cast<float>(output);
  const float ro = static_cast<float>(ref_output);
  const float hw = static_cast<float>(total_weight);
  const float rw = static_cast<float>(ref_total_weight);
  if (std::fabs(ho - ro) > nll_tolerance<scalar_t>(ro) ||
      std::fabs(hw - rw) > nll_tolerance<scalar_t>(rw)) {
    printf("%f %f %f %f\n", ho, ro, hw, rw);
    return false;
  }
  return true;
}

template <typename scalar_t, typename index_t>
void nll_cpu_reference(
    scalar_t* output,
    scalar_t* total_weight,
    const scalar_t* input,
    const index_t* target,
    const scalar_t* weights,
    bool size_average,
    int64_t nframe,
    int64_t kdim,
    int64_t ignore_index)
{
  if constexpr (sizeof(scalar_t) == sizeof(float)) {
    reference<scalar_t, double, index_t>(
        output, total_weight, input, target, weights,
        size_average, nframe, kdim, ignore_index);
  } else {
    reference<scalar_t, scalar_t, index_t>(
        output, total_weight, input, target, weights,
        size_average, nframe, kdim, ignore_index);
  }
}
