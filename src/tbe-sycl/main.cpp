// SYCL kernels inspired by FBGEMM's BSD-3-Clause licensed public
// SplitTableBatchedEmbeddingBags implementation (https://github.com/pytorch/FBGEMM).
// This dependency-free implementation uses the same central optimization idea:
// one subgroup per pooled row and vectorized processing for common D.

#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <vector>

#include "tbe_common.h"

// Host subgroup size. The kernel requests this width, matching warpSize on CUDA/HIP.
inline int tbe_grid_lanes(const sycl::device& device) {
  const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
  const auto largest = std::max_element(sizes.begin(), sizes.end());
  const int lanes = largest == sizes.end() ? 0 : *largest;
  if (lanes <= 0 || tbe_block_threads % lanes != 0) {
    std::fprintf(stderr, "Unsupported subgroup size %d\n", lanes);
    std::exit(2);
  }
  return lanes;
}

inline void tbe_atomic_add(sycl::half* addr, sycl::half value) {
  sycl::atomic_ref<
      sycl::half,
      sycl::memory_order::relaxed,
      sycl::memory_scope::device,
      sycl::access::address_space::global_space>(*addr)
      .fetch_add(value);
}

// One subgroup owns one pooled row. D of 64, 128, 192, or 256 loads four FP16
// values, accumulates them in FP32, and stores FP16. D of 0 is the scalar path.
template <int D, int SG>
sycl::event tbe_launch_forward_fixed(
    sycl::queue& q,
    const TbeHalf* weights,
    const int* indices,
    TbeHalf* output,
    int bags,
    int batch,
    int rows,
    int bag_size,
    int dim) {
  const size_t threads = tbe_block_threads;
  const size_t blocks = tbe_grid_blocks(bags, SG);
  return q.parallel_for(
      sycl::nd_range<1>(blocks * threads, threads),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int lane = it.get_local_id(0) % SG;
        const int warp_in_block = it.get_local_id(0) / SG;
        const int warps_per_block = it.get_local_range(0) / SG;
        const int bag_id = it.get_group(0) * warps_per_block + warp_in_block;
        if (bag_id >= bags) return;
        const size_t bag = bag_id;
        const size_t table = bag / batch;
        const size_t index_base = bag * bag_size;
        const size_t output_base = bag * dim;
        using half4 = sycl::vec<sycl::half, 4>;
        const sycl::half* w16 = reinterpret_cast<const sycl::half*>(weights);
        sycl::half* o16 = reinterpret_cast<sycl::half*>(output);

        if constexpr (D != 0) {
          half4* out4 = reinterpret_cast<half4*>(o16 + output_base);
          for (int v = lane; v < D / 4; v += SG) {
            sycl::vec<float, 4> sum(0.f, 0.f, 0.f, 0.f);
            for (int l = 0; l < bag_size; ++l) {
              const int row = indices[index_base + l];
              const size_t w = (table * rows + row) * D;
              const half4 x = reinterpret_cast<const half4*>(w16 + w)[v];
              sum[0] += float(x[0]);
              sum[1] += float(x[1]);
              sum[2] += float(x[2]);
              sum[3] += float(x[3]);
            }
            out4[v] = half4(sycl::half(sum[0]), sycl::half(sum[1]), sycl::half(sum[2]),
                            sycl::half(sum[3]));
          }
        } else {
          for (int d = lane; d < dim; d += SG) {
            float sum = 0.f;
            for (int l = 0; l < bag_size; ++l) {
              const int row = indices[index_base + l];
              const size_t w = (table * rows + row) * dim;
              sum += float(w16[w + d]);
            }
            o16[output_base + d] = sycl::half(sum);
          }
        }
      });
}

// Scatter-add the pooled gradient into every looked-up row. Repeated indices
// in one bag each issue their own atomic; there is no run-length cache.
template <int D, int SG>
sycl::event tbe_launch_backward_fixed(
    sycl::queue& q,
    const int* indices,
    const TbeHalf* grad_output,
    TbeHalf* grad_weight,
    int bags,
    int batch,
    int rows,
    int bag_size,
    int dim) {
  const size_t threads = tbe_block_threads;
  const size_t blocks = tbe_grid_blocks(bags, SG);
  return q.parallel_for(
      sycl::nd_range<1>(blocks * threads, threads),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int lane = it.get_local_id(0) % SG;
        const int warp_in_block = it.get_local_id(0) / SG;
        const int warps_per_block = it.get_local_range(0) / SG;
        const int bag_id = it.get_group(0) * warps_per_block + warp_in_block;
        if (bag_id >= bags) return;
        const size_t bag = bag_id;
        const size_t table = bag / batch;
        const size_t index_base = bag * bag_size;
        const size_t grad_base = bag * dim;
        using half4 = sycl::vec<sycl::half, 4>;
        const sycl::half* g16 = reinterpret_cast<const sycl::half*>(grad_output);
        sycl::half* gw16 = reinterpret_cast<sycl::half*>(grad_weight);

        if constexpr (D != 0) {
          const half4* grad4 = reinterpret_cast<const half4*>(g16 + grad_base);
          for (int v = lane; v < D / 4; v += SG) {
            const half4 g = grad4[v];
            for (int l = 0; l < bag_size; ++l) {
              const int row = indices[index_base + l];
              const size_t w = (table * rows + row) * D + 4 * v;
              tbe_atomic_add(gw16 + w, g[0]);
              tbe_atomic_add(gw16 + w + 1, g[1]);
              tbe_atomic_add(gw16 + w + 2, g[2]);
              tbe_atomic_add(gw16 + w + 3, g[3]);
            }
          }
        } else {
          for (int d = lane; d < dim; d += SG) {
            const sycl::half g = g16[grad_base + d];
            for (int l = 0; l < bag_size; ++l) {
              const int row = indices[index_base + l];
              const size_t w = (table * rows + row) * dim + d;
              tbe_atomic_add(gw16 + w, g);
            }
          }
        }
      });
}

template <int SG>
sycl::event tbe_launch_forward_sg(
    sycl::queue& q, const TBEProblem& p, const TbeHalf* w, const int* i, TbeHalf* o) {
  switch (p.dim) {
    case 64:
      return tbe_launch_forward_fixed<64, SG>(
          q, w, i, o, p.bags(), p.batch, p.rows, p.bag, p.dim);
    case 128:
      return tbe_launch_forward_fixed<128, SG>(
          q, w, i, o, p.bags(), p.batch, p.rows, p.bag, p.dim);
    case 192:
      return tbe_launch_forward_fixed<192, SG>(
          q, w, i, o, p.bags(), p.batch, p.rows, p.bag, p.dim);
    case 256:
      return tbe_launch_forward_fixed<256, SG>(
          q, w, i, o, p.bags(), p.batch, p.rows, p.bag, p.dim);
    default:
      return tbe_launch_forward_fixed<0, SG>(
          q, w, i, o, p.bags(), p.batch, p.rows, p.bag, p.dim);
  }
}

template <int SG>
sycl::event tbe_launch_backward_sg(
    sycl::queue& q, const TBEProblem& p, const int* i, const TbeHalf* go, TbeHalf* gw) {
  switch (p.dim) {
    case 64:
      return tbe_launch_backward_fixed<64, SG>(
          q, i, go, gw, p.bags(), p.batch, p.rows, p.bag, p.dim);
    case 128:
      return tbe_launch_backward_fixed<128, SG>(
          q, i, go, gw, p.bags(), p.batch, p.rows, p.bag, p.dim);
    case 192:
      return tbe_launch_backward_fixed<192, SG>(
          q, i, go, gw, p.bags(), p.batch, p.rows, p.bag, p.dim);
    case 256:
      return tbe_launch_backward_fixed<256, SG>(
          q, i, go, gw, p.bags(), p.batch, p.rows, p.bag, p.dim);
    default:
      return tbe_launch_backward_fixed<0, SG>(
          q, i, go, gw, p.bags(), p.batch, p.rows, p.bag, p.dim);
  }
}

// Select the vectorized kernel for the published embedding widths.
inline sycl::event tbe_launch_forward(
    sycl::queue& q, const TBEProblem& p, const TbeHalf* w, const int* i, TbeHalf* o, int lanes) {
  switch (lanes) {
    case 64: return tbe_launch_forward_sg<64>(q, p, w, i, o);
    case 32: return tbe_launch_forward_sg<32>(q, p, w, i, o);
    case 16: return tbe_launch_forward_sg<16>(q, p, w, i, o);
    case 8: return tbe_launch_forward_sg<8>(q, p, w, i, o);
    default:
      std::fprintf(stderr, "Unsupported subgroup size %d\n", lanes);
      std::exit(2);
  }
}

// Select the matching backward kernel for the same widths.
inline sycl::event tbe_launch_backward(
    sycl::queue& q,
    const TBEProblem& p,
    const int* i,
    const TbeHalf* go,
    TbeHalf* gw,
    int lanes) {
  switch (lanes) {
    case 64: return tbe_launch_backward_sg<64>(q, p, i, go, gw);
    case 32: return tbe_launch_backward_sg<32>(q, p, i, go, gw);
    case 16: return tbe_launch_backward_sg<16>(q, p, i, go, gw);
    case 8: return tbe_launch_backward_sg<8>(q, p, i, go, gw);
    default:
      std::fprintf(stderr, "Unsupported subgroup size %d\n", lanes);
      std::exit(2);
  }
}

// Check one cleared forward and backward pass, then time `repeat` launches.
// Timed backward does not clear grad_weight between repeats, so later repeats
// accumulate into rows that stay resident. PASS refers to the cleared pass.
int main(int argc, char** argv) {
  TBEProblem p;
  if (!tbe_parse(argc, argv, p)) return 1;
  sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());
  const int lanes = tbe_grid_lanes(q.get_device());
  tbe_print_problem(p, "SYCL");

  std::vector<TbeHalf> weights, grad_output;
  std::vector<int> indices;
  tbe_initialize(p, weights, indices, grad_output);
  std::vector<TbeHalf> output(p.output_count()), output_ref(p.output_count());
  std::vector<TbeHalf> grad_weight(p.weight_count()), grad_weight_ref(p.weight_count());
  tbe_forward_reference(p, weights.data(), indices.data(), output_ref.data());
  tbe_backward_reference(
      p, indices.data(), grad_output.data(), grad_weight_ref.data());

  TbeHalf* dw = sycl::malloc_device<TbeHalf>(weights.size(), q);
  int* di = sycl::malloc_device<int>(indices.size(), q);
  TbeHalf* dout = sycl::malloc_device<TbeHalf>(output.size(), q);
  TbeHalf* dgo = sycl::malloc_device<TbeHalf>(grad_output.size(), q);
  TbeHalf* dgw = sycl::malloc_device<TbeHalf>(grad_weight.size(), q);
  q.copy(weights.data(), dw, weights.size());
  q.copy(indices.data(), di, indices.size());
  q.copy(grad_output.data(), dgo, grad_output.size());

  tbe_launch_forward(q, p, dw, di, dout, lanes);
  q.memset(dgw, 0, grad_weight.size() * sizeof(TbeHalf));
  tbe_launch_backward(q, p, di, dgo, dgw, lanes);
  q.copy(dout, output.data(), output.size());
  q.copy(dgw, grad_weight.data(), grad_weight.size()).wait();
  const bool forward_ok = tbe_check("Forward", output, output_ref);
  const bool backward_ok = tbe_check("Backward", grad_weight, grad_weight_ref);

  for (int k = 0; k < 100; ++k) tbe_launch_forward(q, p, dw, di, dout, lanes);
  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int k = 0; k < p.repeat; ++k) tbe_launch_forward(q, p, dw, di, dout, lanes);
  q.wait();
  auto end = std::chrono::steady_clock::now();
  const double forward_us =
      std::chrono::duration<double, std::micro>(end - start).count() / p.repeat;

  q.memset(dgw, 0, grad_weight.size() * sizeof(TbeHalf));
  for (int k = 0; k < 100; ++k) tbe_launch_backward(q, p, di, dgo, dgw, lanes);
  q.wait();
  start = std::chrono::steady_clock::now();
  for (int k = 0; k < p.repeat; ++k) tbe_launch_backward(q, p, di, dgo, dgw, lanes);
  q.wait();
  end = std::chrono::steady_clock::now();
  const double backward_us =
      std::chrono::duration<double, std::micro>(end - start).count() / p.repeat;
  std::printf("Average execution time of forward kernel: %.3f (us)\n", forward_us);
  std::printf("Average execution time of backward kernel: %.3f (us)\n", backward_us);

  sycl::free(dw, q);
  sycl::free(di, q);
  sycl::free(dout, q);
  sycl::free(dgo, q);
  sycl::free(dgw, q);
  return forward_ok && backward_ok ? 0 : 3;
}
