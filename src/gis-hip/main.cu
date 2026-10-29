#include <chrono>
#include <climits>
#include <cstdio>
#include <vector>

#include <hip/hip_runtime.h>

#include "gis_common.hpp"

#define HIP_CHECK(call)                                                       \
  do {                                                                         \
    const hipError_t error = (call);                                          \
    if (error != hipSuccess) {                                                \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                  \
                   hipGetErrorString(error));                                 \
      return 1;                                                                \
    }                                                                          \
  } while (false)

// Gather selected rows. Each logical warp copies one 32-column tile of one
// row group. The read-only cache load matches FBGEMM's LDG on this path.
__global__ void gis_forward(const float* __restrict__ input,
                            float* __restrict__ output,
                            const int* __restrict__ indices,
                            const int* __restrict__ dims,
                            const int64_t* __restrict__ input_offsets,
                            const int64_t* __restrict__ work_offsets,
                            const int64_t* __restrict__ indices_offsets,
                            const int64_t* __restrict__ worker_offsets,
                            int tables, int work_rows, int64_t total_workers) {
  const int64_t block_threads = blockDim.x;
  const int64_t grid_blocks = gridDim.x;
  const int64_t block = blockIdx.x;
  const int64_t stride = grid_blocks * block_threads / kWarpLanes;
  const int64_t thread = block * block_threads + threadIdx.x;
  const int lane = thread % kWarpLanes;
  for (int64_t worker = thread / kWarpLanes; worker < total_workers;
       worker += stride) {
    const GisLane l = gis_locate(worker, lane, kForwardSteps, worker_offsets,
                                 dims, tables);
    if (!l.active) continue;
    const float* in = input + input_offsets[l.table] + l.col;
    float* out = output + work_offsets[l.table] + l.col;
    const int* idx = indices + indices_offsets[l.table];
    for (int s = 0; s < kForwardSteps; ++s) {
      const int64_t group = l.first_group + s;
      const int64_t row = group * l.rows_per_warp + l.sub_row;
      if (row >= work_rows) break;
      const int64_t index = idx[row];
      out[row * l.dim] = __ldg(in + index * l.dim);
    }
  }
}

// Scatter-add one gradient row per selected index. A lane adds consecutive
// contributions to the same destination in a register and issues one atomic
// add when the index changes. This merges hits only inside its kBackwardSteps
// window, so a production run longer than that window still splits.
__global__ void gis_backward(const float* __restrict__ grad_output,
                             float* __restrict__ input_grad,
                             const int* __restrict__ indices,
                             const int* __restrict__ dims,
                             const int64_t* __restrict__ input_offsets,
                             const int64_t* __restrict__ work_offsets,
                             const int64_t* __restrict__ indices_offsets,
                             const int64_t* __restrict__ worker_offsets,
                             int tables, int work_rows, int64_t total_workers) {
  const int64_t block_threads = blockDim.x;
  const int64_t grid_blocks = gridDim.x;
  const int64_t block = blockIdx.x;
  const int64_t stride = grid_blocks * block_threads / kWarpLanes;
  const int64_t thread = block * block_threads + threadIdx.x;
  const int lane = thread % kWarpLanes;
  for (int64_t worker = thread / kWarpLanes; worker < total_workers;
       worker += stride) {
    const GisLane l = gis_locate(worker, lane, kBackwardSteps, worker_offsets,
                                 dims, tables);
    if (!l.active) continue;
    const float* go = grad_output + work_offsets[l.table] + l.col;
    float* grad = input_grad + input_offsets[l.table] + l.col;
    const int* idx = indices + indices_offsets[l.table];
    int64_t cached_index = -1;
    float cached_sum = 0.0f;
    for (int s = 0; s < kBackwardSteps; ++s) {
      const int64_t group = l.first_group + s;
      const int64_t row = group * l.rows_per_warp + l.sub_row;
      if (row >= work_rows) break;
      const int64_t index = idx[row];
      if (index != cached_index) {
        if (cached_index >= 0)
          atomicAdd(grad + cached_index * l.dim, cached_sum);
        cached_index = index;
        cached_sum = 0.0f;
      }
      cached_sum += go[row * l.dim];
    }
    if (cached_index >= 0) atomicAdd(grad + cached_index * l.dim, cached_sum);
  }
}

// Blocks required to give every logical-warp lane one thread. The kernel
// grid-strides when the uncapped launch would exceed INT_MAX blocks.
static int grid_size(int64_t workers) {
  const int64_t blocks =
      (workers * kWarpLanes + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks > INT_MAX) return INT_MAX;
  return blocks;
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
  const size_t offsets_bytes = (tables + 1) * sizeof(int64_t);

  float *d_input, *d_grad_output, *d_output, *d_input_grad;
  int *d_indices, *d_dims;
  int64_t *d_input_offsets, *d_work_offsets, *d_indices_offsets,
      *d_forward_workers, *d_backward_workers;
  HIP_CHECK(hipMalloc(&d_input, input_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_input_grad, input_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_grad_output, work_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_output, work_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_indices, p.indices.size() * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_dims, tables * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_input_offsets, offsets_bytes));
  HIP_CHECK(hipMalloc(&d_work_offsets, offsets_bytes));
  HIP_CHECK(hipMalloc(&d_indices_offsets, offsets_bytes));
  HIP_CHECK(hipMalloc(&d_forward_workers, offsets_bytes));
  HIP_CHECK(hipMalloc(&d_backward_workers, offsets_bytes));
  HIP_CHECK(hipMemcpy(d_input, p.input.data(), input_elements * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_grad_output, p.grad_output.data(), work_elements * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_indices, p.indices.data(), p.indices.size() * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_dims, p.dims.data(), tables * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_input_offsets, p.input_offsets.data(), offsets_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_work_offsets, p.work_offsets.data(), offsets_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_indices_offsets, p.indices_offsets.data(), offsets_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_forward_workers, p.forward_workers.data(), offsets_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_backward_workers, p.backward_workers.data(), offsets_bytes, hipMemcpyHostToDevice));

  const int forward_blocks = grid_size(forward_workers);
  const int backward_blocks = grid_size(backward_workers);
  auto forward = [&] {
    gis_forward<<<forward_blocks, kThreadsPerBlock>>>(d_input, d_output,
        d_indices, d_dims, d_input_offsets, d_work_offsets, d_indices_offsets,
        d_forward_workers, tables, work_rows, forward_workers);
  };
  auto backward = [&] {
    gis_backward<<<backward_blocks, kThreadsPerBlock>>>(d_grad_output,
        d_input_grad, d_indices, d_dims, d_input_offsets, d_work_offsets,
        d_indices_offsets, d_backward_workers, tables, work_rows,
        backward_workers);
  };

  forward();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemset(d_input_grad, 0, input_elements * sizeof(float)));
  backward();
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> actual_forward(work_elements), actual_backward(input_elements);
  HIP_CHECK(hipMemcpy(actual_forward.data(), d_output, work_elements * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(actual_backward.data(), d_input_grad, input_elements * sizeof(float), hipMemcpyDeviceToHost));
  const bool valid =
      gis_check(actual_forward, expected_forward, 0.0f, "forward") &&
      gis_check(actual_backward, expected_backward, 1e-3f, "backward");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) forward();
  HIP_CHECK(hipDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  std::printf("Average execution time of GIS forward: %.3f (us)\n",
      std::chrono::duration<double, std::micro>(end - start).count() / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) backward();
  HIP_CHECK(hipDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  std::printf("Average execution time of GIS backward: %.3f (us)\n",
      std::chrono::duration<double, std::micro>(end - start).count() / repeat);

  std::printf("%s\n", valid ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_input)); HIP_CHECK(hipFree(d_input_grad));
  HIP_CHECK(hipFree(d_grad_output)); HIP_CHECK(hipFree(d_output));
  HIP_CHECK(hipFree(d_indices)); HIP_CHECK(hipFree(d_dims));
  HIP_CHECK(hipFree(d_input_offsets)); HIP_CHECK(hipFree(d_work_offsets));
  HIP_CHECK(hipFree(d_indices_offsets)); HIP_CHECK(hipFree(d_forward_workers));
  HIP_CHECK(hipFree(d_backward_workers));
  return valid ? 0 : 1;
}
