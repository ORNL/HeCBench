#include <chrono>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <thrust/tuple.h>

#define C10_WARP_SIZE 32
#define C10_HOST_DEVICE __host__ __device__

#include "Array.h"
#include "FunctionTraits.h"
#include <cstdlib>

#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__
#endif

constexpr int num_threads() {
  return C10_WARP_SIZE * 4;
}

constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

template<int arg_index>
struct unroll_load_helper {
  template <typename args_t, typename policy_t, typename offset_t, typename loader_t>
  static __device__ void apply(policy_t &self, args_t *args, offset_t offset, loader_t loader, int j, int num_outputs) {
    using arg_t = std::tuple_element_t<arg_index, args_t>;
    // `data` hold the data_ptr for tensors [output, input0, input1, ...], so we
    // need a+1 offset to get the input
    std::get<arg_index>(args[j]) = loader.template load<arg_t>(self.data[arg_index + num_outputs], offset[arg_index], arg_index);
  }
};

template <int current>
struct multi_outputs_store_helper {
  template<int ntensors, int num_outputs, typename ...Args>
  C10_HOST_DEVICE static void apply(
      Array<char*, ntensors> data,
      Array<uint32_t, num_outputs> offsets,
      thrust::tuple<Args...> ret) {
    using T = typename thrust::tuple_element<current, thrust::tuple<Args...>>::type;
    T *to = reinterpret_cast<T *>(data[current]) + offsets[current];
    *to = thrust::get<current>(ret);
  }
};

// What does the `static_unroll` do?
//
// We want to do something like:
//
//    using args_t = typename traits::ArgsTuple;
//    args_t args;
//    #pragma unroll
//    for (int i = 0; i < traits::arity; i++) {
//      std::get<i>(args) = ....
//    }
//
// but unfortunately the above code does not work because
// the template argument has to be a compile time constant
// so `static_unroll` is created to simulate `#pragma unroll`
// using template metaprogramming.

template<template<int i> typename func, int end, int current=0>
struct static_unroll {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int i> typename func, int end>
struct static_unroll<func, end, end> {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args... args) {}
};

template <typename T>
struct LoadImpl {
  C10_HOST_DEVICE static T apply(const void* src) {
    return *reinterpret_cast<const T*>(src);
  }
};

template <>
struct LoadImpl<bool> {
  C10_HOST_DEVICE static bool apply(const void* src) {
    static_assert(sizeof(bool) == sizeof(char));
    // NOTE: [Loading boolean values]
    // Protect against invalid boolean values by loading as a byte
    // first, then converting to bool (see gh-54789).
    return *reinterpret_cast<const unsigned char*>(src);
  }
};

template <typename T>
C10_HOST_DEVICE T load_impl(const void* src) {
  return LoadImpl<T>::apply(src);
}

template <typename scalar_t>
C10_HOST_DEVICE scalar_t load_impl(const scalar_t* src) {
  return LoadImpl<scalar_t>::apply(src);
}

struct LoadWithoutCast {
  template<typename scalar_t>
  __device__ scalar_t load(char *base_ptr, uint32_t offset, int arg) {
    return load_impl(reinterpret_cast<scalar_t *>(base_ptr) + offset);
  }
};

template <typename data_t, typename inp_calc_t, typename out_calc_t, int num_outputs>
struct multi_outputs_unroll {
  //multi_outputs_unroll struct members and check_inbounds and load methods are copypasted from unroll struct
  //we don't use inheritance because of compiler bug in cuda 10.2+
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  LoadWithoutCast loader;

  __device__ multi_outputs_unroll(data_t data, int remaining, inp_calc_t ic, out_calc_t oc):
    data(data), remaining(remaining), input_offset_calculator(ic), output_offset_calculator(oc) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    return ((threadIdx.x  + thread_work_elem*num_threads()) < remaining);
  }

  template<typename args_t>
  __device__ inline void load(args_t *args, int idx) {
    constexpr int arity = std::tuple_size<args_t>::value;
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size(); i++) {
      if (thread_idx >= remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size() * idx;
      auto offset = input_offset_calculator.get(linear_idx);
      static_unroll<unroll_load_helper, arity>::with_args(*this, args, offset, loader, i, num_outputs);
      thread_idx += num_threads();
    }
  }

  template <typename return_t>
  __device__ inline void store(return_t *from, int idx) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size(); i++) {
      if (thread_idx >= this->remaining) {
        return;
      }
      int linear_idx = thread_idx + block_work_size() * idx;
      auto offsets = this->output_offset_calculator.get(linear_idx);
      static_unroll<multi_outputs_store_helper, num_outputs>::with_args(this->data, offsets, from[i]);
      thread_idx += num_threads();
    }
  }
};

template <class F, class Tuple, std::size_t... INDEX>
// GCC/Clang need the decltype() return type
C10_HOST_DEVICE constexpr decltype(auto) apply_impl(
    F&& f,
    Tuple&& t,
    std::index_sequence<INDEX...>)
{
  return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}

template <class F, class Tuple>
C10_HOST_DEVICE constexpr decltype(auto) guts_apply(F&& f, Tuple&& t) {
  return apply_impl(
      std::forward<F>(f),
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}


template<typename func_t, typename policy_t>
__device__ inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;

  int idx = blockIdx.x;

  return_t results[thread_work_size()];
  args_t args[thread_work_size()];

  // load
  policy.load(args, idx);

  // compute
  #pragma unroll
  for (int i = 0; i < thread_work_size(); i++) {
    if (policy.check_inbounds(i)) {
      results[i] = guts_apply(f, args[i]);
    }
  }

  // store
  policy.store(results, idx);
}

template <int num_outputs, typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
__global__ void unrolled_elementwise_kernel_for_multi_outputs(int N, func_t f, array_t data, inp_calc_t ic, out_calc_t oc) {
  int remaining = N - block_work_size() * blockIdx.x;
  elementwise_kernel_helper(f, multi_outputs_unroll<array_t, inp_calc_t, out_calc_t, num_outputs>(data, remaining, ic, oc));
}

template <int num_outputs, typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
static inline void launch_unrolled_kernel_for_multi_outputs(int64_t N, const func_t& f, array_t data, inp_calc_t ic, out_calc_t oc) {
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  unrolled_elementwise_kernel_for_multi_outputs<num_outputs, func_t, array_t><<<grid, num_threads()>>>(N, f, data, ic, oc);
}

template <int NARGS, typename index_t = uint32_t>
struct TrivialOffsetCalculator {
  // The offset for each argument. Wrapper around fixed-size array.
  // The offsets are in # of elements, not in bytes.
  // On CUDA, zero sized array is not allowed, so when we are handling nullary
  // operators, we need to create a size 1 offset to avoid compiler failure.
  // This size 1 offset is just a placeholder, and we will not use it.
  using offset_type = Array<index_t, std::max<int>(NARGS, 1)>;

  C10_HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;
    #pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = linear_idx;
    }
    return offsets;
  }
};

template <typename func_t>
void gpu_kernel_multiple_outputs_impl(const int repeat, const func_t& f) {
  constexpr int num_outputs = 2;
  constexpr int num_inputs = 4;
  constexpr int ntensors = num_outputs + num_inputs;

  int64_t numel = block_work_size() * 10000;
  printf("Number of elements: %zu\n", numel);

  uint64_t size = numel * sizeof(float);
  
  float *h_x1 = (float*) malloc (size);
  float *h_x2 = (float*) malloc (size);
  float *h_cos = (float*) malloc (size);
  float *h_sin = (float*) malloc (size);
  float *h_o1 = (float*) malloc (size);
  float *h_o2 = (float*) malloc (size);
  for (int64_t i = 0; i < numel; i++) {
    h_x1[i] = 1.f * (i+1) / numel;
    h_x2[i] = 1.f * (i+1) / numel;
    h_cos[i] = cosf(i / powf(10000, i / numel)); 
    h_sin[i] = sinf(i / powf(10000, i / numel));
  }
  
  float *d_x1, *d_x2, *d_cos, *d_sin, *d_o1, *d_o2;
  hipMalloc((void**)&d_x1, size); 
  hipMalloc((void**)&d_x2, size);
  hipMalloc((void**)&d_cos, size);
  hipMalloc((void**)&d_sin, size);
  hipMalloc((void**)&d_o1, size);
  hipMalloc((void**)&d_o2, size);

  hipMemcpy(d_x1, h_x1, size, hipMemcpyHostToDevice);
  hipMemcpy(d_x2, h_x2, size, hipMemcpyHostToDevice);
  hipMemcpy(d_cos, h_cos, size, hipMemcpyHostToDevice);
  hipMemcpy(d_sin, h_sin, size, hipMemcpyHostToDevice);

  Array<char*, ntensors> data;
  data[0] = (char*)d_o1;
  data[1] = (char*)d_o2;
  data[2] = (char*)d_x1;
  data[3] = (char*)d_x2;
  data[4] = (char*)d_cos;
  data[5] = (char*)d_sin;

  auto input_calc = TrivialOffsetCalculator<num_inputs>();
  auto output_calc = TrivialOffsetCalculator<num_outputs>();

  printf("Number of blocks: %zu, block size: %d\n", 
         (numel + block_work_size() - 1) / block_work_size(), num_threads());

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    launch_unrolled_kernel_for_multi_outputs<num_outputs>(numel, f, data, input_calc, output_calc);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);

  hipMemcpy(h_o1, d_o1, size, hipMemcpyDeviceToHost);
  hipMemcpy(h_o2, d_o2, size, hipMemcpyDeviceToHost);
  bool ok = true;
  for (int64_t i = 0; i < numel; i++) {
    float r1 = float(h_x1[i]) * float(h_cos[i]) - float(h_x2[i]) * float(h_sin[i]);
    float r2 = float(h_x1[i]) * float(h_sin[i]) + float(h_x2[i]) * float(h_cos[i]);
    if ((r1 - h_o1[i]) > 1e-3f || (r2 - h_o2[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
  hipFree(d_x1);
  hipFree(d_x2);
  hipFree(d_cos);
  hipFree(d_sin);
  hipFree(d_o1);
  hipFree(d_o2);
  free(h_x1);
  free(h_x2);
  free(h_cos);
  free(h_sin);
  free(h_o1);
  free(h_o2);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  typedef float scalar_t;

  gpu_kernel_multiple_outputs_impl(repeat,
                [] GPU_LAMBDA (scalar_t x1, scalar_t x2, scalar_t cos,
                               scalar_t sin) -> thrust::tuple<scalar_t, scalar_t> {
                scalar_t out1 = float(x1) * float(cos) - float(x2) * float(sin);
                scalar_t out2 = float(x1) * float(sin) + float(x2) * float(cos);
                return {out1, out2};
   });
}
