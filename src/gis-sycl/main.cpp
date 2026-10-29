#include <chrono>
#include <cstdio>
#include <vector>

#include <sycl/sycl.hpp>

#include "gis_common.hpp"

// Verify one forward and one cleared backward pass, then time repeat launches.
// The timed backward loop does not clear its output, so later iterations keep
// updating the same resident rows.
int main(int argc, char** argv) {
  int tables, rows, dim, work_rows, repeat, max_run;
  if (!gis_parse_args(argc, argv, tables, rows, dim, work_rows, repeat, max_run))
    return 1;
  GisProblem p(tables, rows, dim, work_rows, max_run);
  gis_print_problem(p);
  std::vector<float> expected_forward, expected_backward;
  gis_forward_reference(p, expected_forward);
  gis_backward_reference(p, expected_backward);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int64_t work_elements = p.work_elements();
  const int64_t input_elements = p.input_elements();
  const int64_t forward_workers = p.forward_workers.back();
  const int64_t backward_workers = p.backward_workers.back();

  float* d_input = sycl::malloc_device<float>(input_elements, q);
  float* d_input_grad = sycl::malloc_device<float>(input_elements, q);
  float* d_grad_output = sycl::malloc_device<float>(work_elements, q);
  float* d_output = sycl::malloc_device<float>(work_elements, q);
  int* d_indices = sycl::malloc_device<int>(p.indices.size(), q);
  int* d_dims = sycl::malloc_device<int>(tables, q);
  int64_t* d_input_offsets = sycl::malloc_device<int64_t>(tables + 1, q);
  int64_t* d_work_offsets = sycl::malloc_device<int64_t>(tables + 1, q);
  int64_t* d_indices_offsets = sycl::malloc_device<int64_t>(tables + 1, q);
  int64_t* d_forward_workers = sycl::malloc_device<int64_t>(tables + 1, q);
  int64_t* d_backward_workers = sycl::malloc_device<int64_t>(tables + 1, q);
  q.copy(p.input.data(), d_input, input_elements);
  q.copy(p.grad_output.data(), d_grad_output, work_elements);
  q.copy(p.indices.data(), d_indices, p.indices.size());
  q.copy(p.dims.data(), d_dims, tables);
  q.copy(p.input_offsets.data(), d_input_offsets, tables + 1);
  q.copy(p.work_offsets.data(), d_work_offsets, tables + 1);
  q.copy(p.indices_offsets.data(), d_indices_offsets, tables + 1);
  q.copy(p.forward_workers.data(), d_forward_workers, tables + 1);
  q.copy(p.backward_workers.data(), d_backward_workers, tables + 1);
  q.wait();

  // One work-item per logical-warp lane, rounded up to a full work-group.
  auto range_for = [](int64_t workers) {
    const size_t blocks =
        (workers * kWarpLanes + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return sycl::nd_range<1>(blocks * kThreadsPerBlock, kThreadsPerBlock);
  };
  const sycl::nd_range<1> forward_range = range_for(forward_workers);
  const sycl::nd_range<1> backward_range = range_for(backward_workers);

  // Gather selected rows. Each logical warp copies one 32-column tile of one
  // row group.
  auto forward = [&] {
    q.parallel_for(forward_range, [=](sycl::nd_item<1> item) {
      const int64_t gid = item.get_global_id(0);
      const int64_t worker = gid / kWarpLanes;
      if (worker >= forward_workers) return;
      const int lane = gid % kWarpLanes;
      const GisLane l = gis_locate(worker, lane, kForwardSteps,
                                   d_forward_workers, d_dims, tables);
      if (!l.active) return;
      const float* in = d_input + d_input_offsets[l.table] + l.col;
      float* out = d_output + d_work_offsets[l.table] + l.col;
      const int* idx = d_indices + d_indices_offsets[l.table];
      for (int s = 0; s < kForwardSteps; ++s) {
        const int64_t group = l.first_group + s;
        const int64_t row = group * l.rows_per_warp + l.sub_row;
        if (row >= work_rows) break;
        const int64_t index = idx[row];
        out[row * l.dim] = in[index * l.dim];
      }
    });
  };

  // Scatter-add gradient rows. Each lane coalesces consecutive hits in one
  // register and atomically flushes when the destination index changes.
  auto backward = [&] {
    q.parallel_for(backward_range, [=](sycl::nd_item<1> item) {
      const int64_t gid = item.get_global_id(0);
      const int64_t worker = gid / kWarpLanes;
      if (worker >= backward_workers) return;
      const int lane = gid % kWarpLanes;
      const GisLane l = gis_locate(worker, lane, kBackwardSteps,
                                   d_backward_workers, d_dims, tables);
      if (!l.active) return;
      const float* go = d_grad_output + d_work_offsets[l.table] + l.col;
      float* grad = d_input_grad + d_input_offsets[l.table] + l.col;
      const int* idx = d_indices + d_indices_offsets[l.table];
      // Add one cached partial sum into the selected embedding row.
      auto flush = [&](int64_t index, float value) {
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(grad[index * l.dim]);
        ref.fetch_add(value);
      };
      int64_t cached_index = -1;
      float cached_sum = 0.0f;
      for (int s = 0; s < kBackwardSteps; ++s) {
        const int64_t group = l.first_group + s;
        const int64_t row = group * l.rows_per_warp + l.sub_row;
        if (row >= work_rows) break;
        const int64_t index = idx[row];
        if (index != cached_index) {
          if (cached_index >= 0) flush(cached_index, cached_sum);
          cached_index = index;
          cached_sum = 0.0f;
        }
        cached_sum += go[row * l.dim];
      }
      if (cached_index >= 0) flush(cached_index, cached_sum);
    });
  };

  forward();
  q.memset(d_input_grad, 0, input_elements * sizeof(float));
  backward();
  q.wait();

  std::vector<float> actual_forward(work_elements), actual_backward(input_elements);
  q.copy(d_output, actual_forward.data(), work_elements);
  q.copy(d_input_grad, actual_backward.data(), input_elements);
  q.wait();
  const bool valid =
      gis_check(actual_forward, expected_forward, 0.0f, "forward") &&
      gis_check(actual_backward, expected_backward, 1e-3f, "backward");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) forward();
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::printf("Average execution time of GIS forward: %.3f (us)\n",
      std::chrono::duration<double, std::micro>(end - start).count() / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) backward();
  q.wait();
  end = std::chrono::steady_clock::now();
  std::printf("Average execution time of GIS backward: %.3f (us)\n",
      std::chrono::duration<double, std::micro>(end - start).count() / repeat);

  std::printf("%s\n", valid ? "PASS" : "FAIL");

  sycl::free(d_input, q); sycl::free(d_input_grad, q);
  sycl::free(d_grad_output, q); sycl::free(d_output, q);
  sycl::free(d_indices, q); sycl::free(d_dims, q);
  sycl::free(d_input_offsets, q); sycl::free(d_work_offsets, q);
  sycl::free(d_indices_offsets, q); sycl::free(d_forward_workers, q);
  sycl::free(d_backward_workers, q);
  return valid ? 0 : 1;
}
