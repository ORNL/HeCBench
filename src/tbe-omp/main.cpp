// Standalone implementation derived from the concepts in FBGEMM's public
// SplitTableBatchedEmbeddingBags code (BSD 3-Clause):
// https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu

#include <omp.h>
#include <chrono>
#include <cstdint>
#include <vector>
#include "tbe_common.h"

// One team owns one pooled row. thread_limit 64 matches a wavefront.
// D of 64, 128, 192, or 256 loads four FP16 values and accumulates them in FP32.
static void forward_omp(
    const TbeHalf* weights,
    const int* indices,
    TbeHalf* output,
    int tables,
    int batch,
    int rows,
    int dim,
    int bag_size) {
  const int bags = tables * batch;
#pragma omp target teams distribute is_device_ptr(weights, indices, output) thread_limit(64)
  for (int bag_id = 0; bag_id < bags; ++bag_id) {
    const size_t bag = bag_id;
    const size_t table = bag / batch;
    const size_t ib = bag * bag_size;
    const size_t ob = bag * dim;
    // A team is the OpenMP analogue of FBGEMM's warp/wavefront-per-row map.
#pragma omp parallel
    {
      const int lane = omp_get_thread_num();
      const int lanes = omp_get_num_threads();
      const _Float16* w16 = reinterpret_cast<const _Float16*>(weights);
      _Float16* o16 = reinterpret_cast<_Float16*>(output);
      if (dim == 64 || dim == 128 || dim == 192 || dim == 256) {
        for (int v = lane; v < dim / 4; v += lanes) {
          float x0 = 0.f, x1 = 0.f, x2 = 0.f, x3 = 0.f;
          for (int l = 0; l < bag_size; ++l) {
            const int row = indices[ib + l];
            const size_t w = (table * rows + row) * dim + 4 * v;
            x0 += float(w16[w]);
            x1 += float(w16[w + 1]);
            x2 += float(w16[w + 2]);
            x3 += float(w16[w + 3]);
          }
          o16[ob + 4 * v] = _Float16(x0);
          o16[ob + 4 * v + 1] = _Float16(x1);
          o16[ob + 4 * v + 2] = _Float16(x2);
          o16[ob + 4 * v + 3] = _Float16(x3);
        }
      } else {
        for (int d = lane; d < dim; d += lanes) {
          float sum = 0.f;
          for (int l = 0; l < bag_size; ++l) {
            const int row = indices[ib + l];
            const size_t w = (table * rows + row) * dim + d;
            sum += float(w16[w]);
          }
          o16[ob + d] = _Float16(sum);
        }
      }
    }
  }
}

// icpx lowers an FP16 atomic to __spirv_AtomicFAddEXT. Level Zero implements
// neither that nor 16-bit integer atomics, so update the binary16 bits with a
// 32-bit compare-and-swap. amdclang++ keeps the native FP16 atomic.
#pragma omp declare target
static void tbe_atomic_add(_Float16* addr, _Float16 value) {
#if defined(__INTEL_LLVM_COMPILER)
  const auto raw = reinterpret_cast<uintptr_t>(addr);
  const bool high = raw & 2u;
  auto* word = reinterpret_cast<unsigned int*>(raw & ~uintptr_t{3});
  unsigned int observed;
  #pragma omp atomic read
  observed = *word;
  for (;;) {
    const unsigned int expected = observed;
    const unsigned short bits = high ? static_cast<unsigned short>(expected >> 16)
                                      : static_cast<unsigned short>(expected);
    _Float16 current;
    __builtin_memcpy(&current, &bits, sizeof(current));
    const _Float16 updated = _Float16(float(current) + float(value));
    unsigned short desired_bits;
    __builtin_memcpy(&desired_bits, &updated, sizeof(desired_bits));
    const unsigned int desired_half = desired_bits;
    const unsigned int desired =
        high ? (expected & 0x0000ffffu) | (desired_half << 16)
             : (expected & 0xffff0000u) | desired_half;
    unsigned int current_word = expected;
  #pragma omp atomic compare capture
    {
      current_word = *word;
      if (*word == expected) { *word = desired; }
    }
    if (current_word == expected) return;
    observed = current_word;
  }
#else

  #pragma omp atomic update
    *addr += value;
#endif
}
#pragma omp end declare target

// Atomic scatter-add. Each bag entry updates its row; repeats are not coalesced.
static void backward_omp(
    const int* indices,
    const TbeHalf* grad_output,
    TbeHalf* grad_weight,
    int tables,
    int batch,
    int rows,
    int dim,
    int bag_size) {
  const int bags = tables * batch;
#pragma omp target teams distribute is_device_ptr(indices, grad_output, grad_weight) thread_limit(64)
  for (int bag_id = 0; bag_id < bags; ++bag_id) {
    const size_t bag = bag_id;
    const size_t table = bag / batch;
    const size_t ib = bag * bag_size;
    const size_t gb = bag * dim;
#pragma omp parallel
    {
      const int lane = omp_get_thread_num();
      const int lanes = omp_get_num_threads();
      const _Float16* g16 = reinterpret_cast<const _Float16*>(grad_output);
      _Float16* gw16 = reinterpret_cast<_Float16*>(grad_weight);
      if (dim == 64 || dim == 128 || dim == 192 || dim == 256) {
        for (int v = lane; v < dim / 4; v += lanes) {
          const _Float16 g0 = g16[gb + 4 * v];
          const _Float16 g1 = g16[gb + 4 * v + 1];
          const _Float16 g2 = g16[gb + 4 * v + 2];
          const _Float16 g3 = g16[gb + 4 * v + 3];
          for (int l = 0; l < bag_size; ++l) {
            const int row = indices[ib + l];
            const size_t w = (table * rows + row) * dim + 4 * v;
            tbe_atomic_add(gw16 + w, g0);
            tbe_atomic_add(gw16 + w + 1, g1);
            tbe_atomic_add(gw16 + w + 2, g2);
            tbe_atomic_add(gw16 + w + 3, g3);
          }
        }
      } else {
        for (int d = lane; d < dim; d += lanes) {
          const _Float16 g = g16[gb + d];
          for (int l = 0; l < bag_size; ++l) {
            const int row = indices[ib + l];
            const size_t w = (table * rows + row) * dim + d;
            tbe_atomic_add(gw16 + w, g);
          }
        }
      }
    }
  }
}

// Zero a device buffer before the checked backward pass and before warmup.
static void clear_omp(TbeHalf* data, size_t count) {
#pragma omp target teams distribute parallel for is_device_ptr(data)
  for (size_t i = 0; i < count; ++i) data[i] = TbeHalf{0};
}

// Check one cleared pass, then time `repeat` launches. Timed backward does not
// clear grad_weight between repeats, so PASS is the cleared pass only.
// Host fallback runs the same teams region when no target device is present.
int main(int argc, char** argv) {
  TBEProblem p;
  if (!tbe_parse(argc, argv, p)) return 1;
  tbe_print_problem(p, "OpenMP");
  std::printf(
      "OpenMP target devices: %d%s\n",
      omp_get_num_devices(),
      omp_get_num_devices() ? "" : " (host fallback)");

  std::vector<TbeHalf> weights, grad_output;
  std::vector<int> indices;
  tbe_initialize(p, weights, indices, grad_output);
  std::vector<TbeHalf> output(p.output_count()), output_ref(p.output_count());
  // The Intel atomic updates a 32-bit word. An odd length needs one padding
  // element so the final low half does not write past the allocation.
  const size_t grad_count = p.weight_count() + (p.weight_count() & 1);
  std::vector<TbeHalf> grad_weight(grad_count), grad_weight_ref(p.weight_count());
  tbe_forward_reference(p, weights.data(), indices.data(), output_ref.data());
  tbe_backward_reference(
      p, indices.data(), grad_output.data(), grad_weight_ref.data());
  const size_t wc = weights.size(), ic = indices.size(), oc = output.size();
  const size_t gc = grad_weight.size();
  TbeHalf* w = weights.data();
  int* idx = indices.data();
  TbeHalf* out = output.data();
  TbeHalf* go = grad_output.data();
  TbeHalf* gw = grad_weight.data();
  bool forward_ok = false, backward_ok = false;
  double forward_us = 0.0, backward_us = 0.0;

#pragma omp target data map(to : w[0:wc], idx[0:ic], go[0:oc]) map(alloc : out[0:oc], gw[0:gc])
  {
#pragma omp target data use_device_ptr(w, idx, go, out, gw)
    {
      forward_omp(w, idx, out, p.tables, p.batch, p.rows, p.dim, p.bag);
      clear_omp(gw, gc);
      backward_omp(idx, go, gw, p.tables, p.batch, p.rows, p.dim, p.bag);
    }
#pragma omp target update from(out[0:oc], gw[0:gc])
    forward_ok = tbe_check("Forward", output, output_ref);
    backward_ok = tbe_check("Backward", grad_weight, grad_weight_ref);

#pragma omp target data use_device_ptr(w, idx, go, out, gw)
    {
      for (int k = 0; k < 100; ++k)
        forward_omp(w, idx, out, p.tables, p.batch, p.rows, p.dim, p.bag);

      auto start = std::chrono::steady_clock::now();
      for (int k = 0; k < p.repeat; ++k)
        forward_omp(w, idx, out, p.tables, p.batch, p.rows, p.dim, p.bag);
      auto end = std::chrono::steady_clock::now();
      forward_us =
          std::chrono::duration<double, std::micro>(end - start).count() / p.repeat;

      clear_omp(gw, gc);
      for (int k = 0; k < 100; ++k)
        backward_omp(idx, go, gw, p.tables, p.batch, p.rows, p.dim, p.bag);

      start = std::chrono::steady_clock::now();
      for (int k = 0; k < p.repeat; ++k)
        backward_omp(idx, go, gw, p.tables, p.batch, p.rows, p.dim, p.bag);
      end = std::chrono::steady_clock::now();
      backward_us =
          std::chrono::duration<double, std::micro>(end - start).count() / p.repeat;
    }
  }
  std::printf("Average execution time of forward kernel: %.3f (us)\n", forward_us);
  std::printf("Average execution time of backward kernel: %.3f (us)\n", backward_us);
  return forward_ok && backward_ok ? 0 : 3;
}
