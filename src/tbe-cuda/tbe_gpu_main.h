#pragma once

#include <chrono>
#include <vector>

#include "tbe_common.h"
#include "tbe_gpu.h"

// Check one cleared forward and backward pass, then time `repeat` launches.
// Timed backward does not clear grad_weight between repeats, so later repeats
// accumulate into rows that stay resident. PASS refers to the cleared pass.
// cudaFree and hipFree are checked by TBE_GPU_CHECK.
inline int tbe_gpu_main(int argc, char** argv, const char* backend) {
  TBEProblem p;
  if (!tbe_parse(argc, argv, p)) return 1;
  tbe_print_problem(p, backend);

  std::vector<TbeHalf> weights, grad_output;
  std::vector<int> indices;
  tbe_initialize(p, weights, indices, grad_output);
  std::vector<TbeHalf> output(p.output_count()), output_ref(p.output_count());
  std::vector<TbeHalf> grad_weight(p.weight_count()), grad_weight_ref(p.weight_count());
  tbe_forward_reference(p, weights.data(), indices.data(), output_ref.data());
  tbe_backward_reference(
      p, indices.data(), grad_output.data(), grad_weight_ref.data());

  TbeHalf *dw = nullptr, *dout = nullptr, *dgo = nullptr, *dgw = nullptr;
  int* di = nullptr;
  TBE_GPU_CHECK(TBE_MALLOC(reinterpret_cast<void**>(&dw), weights.size() * sizeof(TbeHalf)));
  TBE_GPU_CHECK(TBE_MALLOC(reinterpret_cast<void**>(&di), indices.size() * sizeof(int)));
  TBE_GPU_CHECK(TBE_MALLOC(reinterpret_cast<void**>(&dout), output.size() * sizeof(TbeHalf)));
  TBE_GPU_CHECK(TBE_MALLOC(reinterpret_cast<void**>(&dgo), grad_output.size() * sizeof(TbeHalf)));
  TBE_GPU_CHECK(TBE_MALLOC(reinterpret_cast<void**>(&dgw), grad_weight.size() * sizeof(TbeHalf)));
  TBE_GPU_CHECK(TBE_MEMCPY(dw, weights.data(), weights.size() * sizeof(TbeHalf), TBE_H2D));
  TBE_GPU_CHECK(TBE_MEMCPY(di, indices.data(), indices.size() * sizeof(int), TBE_H2D));
  TBE_GPU_CHECK(
      TBE_MEMCPY(dgo, grad_output.data(), grad_output.size() * sizeof(TbeHalf), TBE_H2D));

  const int lanes = tbe_grid_lanes();
  tbe_launch_forward(p, dw, di, dout, lanes);
  TBE_GPU_CHECK(TBE_MEMSET(dgw, 0, grad_weight.size() * sizeof(TbeHalf)));
  tbe_launch_backward(p, di, dgo, dgw, lanes);
  TBE_GPU_CHECK(TBE_SYNC());
  TBE_GPU_CHECK(TBE_LAST_ERROR());
  TBE_GPU_CHECK(TBE_MEMCPY(
      output.data(), dout, output.size() * sizeof(TbeHalf), TBE_D2H));
  TBE_GPU_CHECK(TBE_MEMCPY(
      grad_weight.data(), dgw, grad_weight.size() * sizeof(TbeHalf), TBE_D2H));
  const bool forward_ok = tbe_check("Forward", output, output_ref);
  const bool backward_ok = tbe_check("Backward", grad_weight, grad_weight_ref);

  for (int k = 0; k < 100; ++k) tbe_launch_forward(p, dw, di, dout, lanes);
  TBE_GPU_CHECK(TBE_SYNC());
  auto start = std::chrono::steady_clock::now();
  for (int k = 0; k < p.repeat; ++k) tbe_launch_forward(p, dw, di, dout, lanes);
  TBE_GPU_CHECK(TBE_SYNC());
  auto end = std::chrono::steady_clock::now();
  const double forward_us =
      std::chrono::duration<double, std::micro>(end - start).count() / p.repeat;

  TBE_GPU_CHECK(TBE_MEMSET(dgw, 0, grad_weight.size() * sizeof(TbeHalf)));
  for (int k = 0; k < 100; ++k) tbe_launch_backward(p, di, dgo, dgw, lanes);
  TBE_GPU_CHECK(TBE_SYNC());
  start = std::chrono::steady_clock::now();
  for (int k = 0; k < p.repeat; ++k) tbe_launch_backward(p, di, dgo, dgw, lanes);
  TBE_GPU_CHECK(TBE_SYNC());
  end = std::chrono::steady_clock::now();
  const double backward_us =
      std::chrono::duration<double, std::micro>(end - start).count() / p.repeat;
  std::printf("Average execution time of forward kernel: %.3f (us)\n", forward_us);
  std::printf("Average execution time of backward kernel: %.3f (us)\n", backward_us);

  TBE_GPU_CHECK(TBE_FREE(dw));
  TBE_GPU_CHECK(TBE_FREE(di));
  TBE_GPU_CHECK(TBE_FREE(dout));
  TBE_GPU_CHECK(TBE_FREE(dgo));
  TBE_GPU_CHECK(TBE_FREE(dgw));
  return forward_ok && backward_ok ? 0 : 3;
}
