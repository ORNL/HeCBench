#include <chrono>
#include <cstdio>
#include <vector>

#include <omp.h>

#include "gis_common.hpp"

// Gather selected rows. One loop iteration is one logical-warp lane, and each
// warp copies one 32-column tile of one row group.
static void gis_forward(const float* __restrict__ src, float* __restrict__ dst,
                        const int* __restrict__ indices, const int* __restrict__ dims,
                        const int64_t* __restrict__ input_offsets,
                        const int64_t* __restrict__ work_offsets,
                        const int64_t* __restrict__ indices_offsets,
                        const int64_t* __restrict__ worker_offsets,
                        int tables, int work_rows, int64_t workers) {
  #pragma omp target teams distribute parallel for thread_limit(kThreadsPerBlock)
  for (int64_t gid = 0; gid < workers * kWarpLanes; ++gid) {
    const int64_t worker = gid / kWarpLanes;
    const int lane = gid % kWarpLanes;
    const GisLane l = gis_locate(worker, lane, kForwardSteps,
                                 worker_offsets, dims, tables);
    if (!l.active) continue;
    const float* in = src + input_offsets[l.table] + l.col;
    float* out = dst + work_offsets[l.table] + l.col;
    const int* idx = indices + indices_offsets[l.table];
    for (int s = 0; s < kForwardSteps; ++s) {
      const int64_t group = l.first_group + s;
      const int64_t row = group * l.rows_per_warp + l.sub_row;
      if (row >= work_rows) break;
      const int64_t index = idx[row];
      out[row * l.dim] = in[index * l.dim];
    }
  }
}

// Scatter-add gradient rows. Each lane coalesces consecutive hits in one
// register and atomically flushes when the destination index changes.
static void gis_backward(const float* __restrict__ src, float* __restrict__ dst,
                        const int* __restrict__ indices, const int* __restrict__ dims,
                        const int64_t* __restrict__ input_offsets,
                        const int64_t* __restrict__ work_offsets,
                        const int64_t* __restrict__ indices_offsets,
                        const int64_t* __restrict__ worker_offsets,
                        int tables, int work_rows, int64_t workers) {
  #pragma omp target teams distribute parallel for thread_limit(kThreadsPerBlock)
  for (int64_t gid = 0; gid < workers * kWarpLanes; ++gid) {
    const int64_t worker = gid / kWarpLanes;
    const int lane = gid % kWarpLanes;
    const GisLane l = gis_locate(worker, lane, kBackwardSteps,
                                 worker_offsets, dims, tables);
    if (!l.active) continue;
    const float* go = src + work_offsets[l.table] + l.col;
    float* grad = dst + input_offsets[l.table] + l.col;
    const int* idx = indices + indices_offsets[l.table];
    int64_t cached_index = -1;
    float cached_sum = 0.0f;
    for (int s = 0; s < kBackwardSteps; ++s) {
      const int64_t group = l.first_group + s;
      const int64_t row = group * l.rows_per_warp + l.sub_row;
      if (row >= work_rows) break;
      const int64_t index = idx[row];
      if (index != cached_index) {
        if (cached_index >= 0) {
          #pragma omp atomic update
          grad[cached_index * l.dim] += cached_sum;
        }
        cached_index = index;
        cached_sum = 0.0f;
      }
      cached_sum += go[row * l.dim];
    }
    if (cached_index >= 0) {
      #pragma omp atomic update
      grad[cached_index * l.dim] += cached_sum;
    }
  }
}

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

  const int64_t work_elements = p.work_elements();
  const int64_t input_elements = p.input_elements();
  const int64_t forward_workers = p.forward_workers.back();
  const int64_t backward_workers = p.backward_workers.back();
  const int64_t num_indices = p.indices.size();

  const float* input = p.input.data();
  const float* grad_output = p.grad_output.data();
  const int* indices = p.indices.data();
  const int* dims = p.dims.data();
  const int64_t* input_offsets = p.input_offsets.data();
  const int64_t* work_offsets = p.work_offsets.data();
  const int64_t* indices_offsets = p.indices_offsets.data();
  const int64_t* forward_offsets = p.forward_workers.data();
  const int64_t* backward_offsets = p.backward_workers.data();
  std::vector<float> actual_forward(work_elements), actual_backward(input_elements);
  float* output = actual_forward.data();
  float* input_grad = actual_backward.data();

  auto forward = [&] {
    gis_forward(input, output, indices, dims, input_offsets, work_offsets,
                indices_offsets, forward_offsets, tables, work_rows,
                forward_workers);
  };
  auto backward = [&] {
    gis_backward(grad_output, input_grad, indices, dims, input_offsets,
                 work_offsets, indices_offsets, backward_offsets, tables,
                 work_rows, backward_workers);
  };

  bool valid;
  #pragma omp target data map(to: input[0:input_elements], \
                                  grad_output[0:work_elements], \
                                  indices[0:num_indices], dims[0:tables], \
                                  input_offsets[0:tables + 1], \
                                  work_offsets[0:tables + 1], \
                                  indices_offsets[0:tables + 1], \
                                  forward_offsets[0:tables + 1], \
                                  backward_offsets[0:tables + 1]) \
                          map(alloc: output[0:work_elements]) \
                          map(to: input_grad[0:input_elements])
  {
    forward();
    backward();
    #pragma omp target update from(output[0:work_elements])
    #pragma omp target update from(input_grad[0:input_elements])
    valid = gis_check(actual_forward, expected_forward, 0.0f, "forward") &&
            gis_check(actual_backward, expected_backward, 1e-3f, "backward");

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) forward();
    auto end = std::chrono::steady_clock::now();
    std::printf("Average execution time of GIS forward: %.3f (us)\n",
        std::chrono::duration<double, std::micro>(end - start).count() / repeat);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) backward();
    end = std::chrono::steady_clock::now();
    std::printf("Average execution time of GIS backward: %.3f (us)\n",
        std::chrono::duration<double, std::micro>(end - start).count() / repeat);
  }

  std::printf("%s\n", valid ? "PASS" : "FAIL");
  return valid ? 0 : 1;
}
