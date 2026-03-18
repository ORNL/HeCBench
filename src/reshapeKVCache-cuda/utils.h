#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#define GPU_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)


enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(
    const Tin& x, const float scale, const __nv_fp8_interpretation_t fp8_type) {
  return x;
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
    #else
  __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                                 __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
    #endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

// float -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, float>(
    const float& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(a / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}


template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E4M3);
  }
  assert(false);
  return {};  // Squash missing return statement warning
}


template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  float scale;

  // copy or convert one element
  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    }
    else {
      dst = scaled_convert<OutT, InT, kv_dt>(src, scale);
    }
  }
};


template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};


template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
struct DefaultVecOp {
  ScaOp scalar_op;

  __device__ __forceinline__ void operator()(
      vec_n_t<OutT, VEC_SIZE>& dst, const vec_n_t<InT, VEC_SIZE>& src) const {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      scalar_op(dst.val[i], src.val[i]);
    }
  }
};

template <int VEC_SIZE, typename InT, typename OutT, typename VecOp,
          typename ScaOp>
__device__ inline void vectorize_with_alignment(
    const InT* in, OutT* out, int len, int tid, int stride,
    VecOp&& vec_op,       // vec_n_t<InT,16> -> vec_n_t<OutT,16>
    ScaOp&& scalar_op) {  // InT -> OutT
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * sizeof(InT);  // eg: 64 B
  uintptr_t addr = reinterpret_cast<uintptr_t>(in);

  // fast path when the whole region is already aligned
  // Note: currently the output is guaranteed to be same as the input, so we
  // don't check it here, comments here just for future reference.
  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int num_vec = len / VEC_SIZE;

    using vin_t = vec_n_t<InT, VEC_SIZE>;
    using vout_t = vec_n_t<OutT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<const vin_t*>(in);
    auto* v_out = reinterpret_cast<vout_t*>(out);

    for (int i = tid; i < num_vec; i += stride) {
      vout_t tmp;
      // Make a local copy of the entire pack
      vin_t src = v_in[i];  // <- encourages a single vector ld
      vec_op(tmp, src);
      v_out[i] = tmp;  // <- encourages a single vector st
    }
    return;
  }

  int misalignment_offset = addr & (WIDTH - 1);       // addr % 64
  int alignment_bytes = WIDTH - misalignment_offset;  // 64 - (addr % 64)
  int prefix_elems = alignment_bytes & (WIDTH - 1);   // handle 64
  prefix_elems /= sizeof(InT);
  prefix_elems = min(prefix_elems, len);  // 0 ≤ prefix < 16

  // 1. prefill the when it is unsafe to vectorize
  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(out[i], in[i]);
  }

  in += prefix_elems;
  out += prefix_elems;
  len -= prefix_elems;

  int num_vec = len / VEC_SIZE;
  using vin_t = vec_n_t<InT, VEC_SIZE>;
  using vout_t = vec_n_t<OutT, VEC_SIZE>;
  auto* v_in = reinterpret_cast<const vin_t*>(in);
  auto* v_out = reinterpret_cast<vout_t*>(out);

  // 2. vectorize the main part
  for (int i = tid; i < num_vec; i += stride) {
    vout_t tmp;
    // Make a local copy of the entire pack
    vin_t src = v_in[i];  // <- encourages a single vector ld
    vec_op(tmp, src);
    v_out[i] = tmp;  // <- encourages a single vector st
  }

  // 3. handle the tail
  int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(out[i], in[i]);
  }
}

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
__device__ __forceinline__ void vectorize_with_alignment(const InT* in,
                                                         OutT* out, int len,
                                                         int tid, int stride,
                                                         ScaOp&& scalar_op) {
  using Vec = DefaultVecOp<VEC_SIZE, InT, OutT, std::decay_t<ScaOp>>;
  vectorize_with_alignment<VEC_SIZE>(in, out, len, tid, stride, Vec{scalar_op},
                                     std::forward<ScaOp>(scalar_op));
}
