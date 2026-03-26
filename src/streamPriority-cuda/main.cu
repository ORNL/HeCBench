#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include "kernels.h"
#include "reference.h"

#define GPU_CHECK(expr)                                                 \
    do { cudaError_t e = (expr);                                        \
         if (e != cudaSuccess) {                                        \
             fprintf(stderr, "CUDA %s:%d  %s\n",                        \
                     __FILE__, __LINE__, cudaGetErrorString(e));        \
             exit(EXIT_FAILURE); } } while (0)

//----------------------------------------------------------------
//   high priority  - forward + backward (blocks training loop)
//   low  priority  - data prefetch for next batch (background)
//----------------------------------------------------------------
void use_case_ml_training(int batches, int lo_pri, int hi_pri) {
  const int N       = 1 << 24;
  const int threads = 256;
  const int blocks  = (N + threads - 1) / threads;

  // high and low priority streams
  cudaStream_t s_train, s_prefetch;
  GPU_CHECK(cudaStreamCreateWithPriority(&s_train,    cudaStreamNonBlocking, hi_pri));
  GPU_CHECK(cudaStreamCreateWithPriority(&s_prefetch, cudaStreamNonBlocking, lo_pri));

  // events for inter-stream synchronization
  cudaEvent_t fwd_done, fetch_done;
  GPU_CHECK(cudaEventCreate(&fwd_done));
  GPU_CHECK(cudaEventCreate(&fetch_done));

  std::vector<float> prefetch_buf (N, 0);
  std::vector<float> weights (N, 1);
  std::vector<float> weights_ref (N, 1); // constant
  std::vector<float> grad_in (N, 1);
  std::vector<float> grad_in_ref (N, 1); // constant
  std::vector<float> act (N);
  std::vector<float> grad_out (N);
  std::vector<float> grad_out_dev (N); // from device

  float *d_act, *d_grad_in, *d_grad_out, *d_prefetch_buf, *d_weights;
  GPU_CHECK(cudaMalloc(&d_act,          N * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_grad_in,      N * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_grad_out,     N * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_prefetch_buf, N * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_weights,      N * sizeof(float)));

  GPU_CHECK(cudaMemset(d_prefetch_buf, 0, N * sizeof(float)));
  GPU_CHECK(cudaMemcpy(d_grad_in, grad_in_ref.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_weights, weights_ref.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  //------------------------------------------------------------
  // reference
  //------------------------------------------------------------
  for (int b = 0; b < batches; b++) {
    forward_pass(prefetch_buf.data(), act.data(), N, 1.0f);
    backward_pass(act.data(), grad_in.data(), grad_out.data(), N);
    sgd_update(weights.data(), grad_out.data(), N, 0.01f);
    data_prefetch(prefetch_buf.data(), N, b + 1);
  }

  //------------------------------------------------------------
  printf("Default stream (no priority)\n");
  //------------------------------------------------------------
  GPU_CHECK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  for (int b = 0; b < batches; b++) {
    forward_pass_kernel <<<blocks, threads>>>(d_prefetch_buf, d_act,      N, 1.0f);
    backward_pass_kernel<<<blocks, threads>>>(d_act,      d_grad_in,  d_grad_out, N);
    sgd_update_kernel   <<<blocks, threads>>>(d_weights,  d_grad_out, N, 0.01f);
    data_prefetch_kernel<<<blocks, threads>>>(d_prefetch_buf, N, b + 1);
  }
  GPU_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time_default = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  GPU_CHECK(cudaMemcpy(grad_out_dev.data(), d_grad_out, N * sizeof(float), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(grad_out_dev[i] - grad_out[i]) > 1e-3f) {
      printf("@%d: %f %f\n", i, grad_out_dev[i] , grad_out[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  //------------------------------------------------------------
  printf("Priority stream\n");
  //------------------------------------------------------------
  GPU_CHECK(cudaMemset(d_prefetch_buf, 0, N * sizeof(float)));
  GPU_CHECK(cudaMemcpy(d_grad_in, grad_in_ref.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_weights, weights_ref.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  GPU_CHECK(cudaDeviceSynchronize());
  start = std::chrono::steady_clock::now();
  for (int b = 0; b < batches; b++) {
    GPU_CHECK(cudaStreamWaitEvent(s_train, fetch_done, 0));
    forward_pass_kernel <<<blocks, threads, 0, s_train>>>(d_prefetch_buf, d_act, N, 1.0f);
    GPU_CHECK(cudaEventRecord(fwd_done, s_train));
    backward_pass_kernel<<<blocks, threads, 0, s_train>>>(d_act, d_grad_in, d_grad_out, N);
    sgd_update_kernel<<<blocks, threads, 0, s_train>>>(d_weights, d_grad_out, N, 0.01f);
    GPU_CHECK(cudaStreamWaitEvent(s_prefetch, fwd_done, 0));
    data_prefetch_kernel<<<blocks, threads, 0, s_prefetch>>>(d_prefetch_buf, N, b + 1);
    GPU_CHECK(cudaEventRecord(fetch_done, s_prefetch));
  }
  GPU_CHECK(cudaStreamSynchronize(s_train)); // wait for s_train stream
  end = std::chrono::steady_clock::now();
  auto time_pri = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  GPU_CHECK(cudaMemcpyAsync(grad_out_dev.data(), d_grad_out, N * sizeof(float), cudaMemcpyDeviceToHost, s_train));
  GPU_CHECK(cudaStreamSynchronize(s_train)); // wait for s_train stream

  ok = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(grad_out_dev[i] - grad_out[i]) > 1e-3f) {
      printf("@%d: %f %f\n", i, grad_out_dev[i] , grad_out[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  printf("  Default  streams: %.2f ms\n", time_default * 1e-6);
  printf("  Priority streams: %.2f ms\n", time_pri * 1e-6);

  GPU_CHECK(cudaStreamDestroy(s_train));
  GPU_CHECK(cudaStreamDestroy(s_prefetch));
  GPU_CHECK(cudaEventDestroy(fwd_done));
  GPU_CHECK(cudaEventDestroy(fetch_done));
  GPU_CHECK(cudaFree(d_act));
  GPU_CHECK(cudaFree(d_grad_in));
  GPU_CHECK(cudaFree(d_grad_out));
  GPU_CHECK(cudaFree(d_prefetch_buf));
  GPU_CHECK(cudaFree(d_weights));
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int lo_pri, hi_pri;
  GPU_CHECK(cudaDeviceGetStreamPriorityRange(&lo_pri, &hi_pri));
  printf("Stream priority range:  least=%d  greatest=%d\n", lo_pri, hi_pri);
  printf("(greatest priority = numerically lowest value = %d)\n\n", hi_pri);

  use_case_ml_training(repeat, lo_pri, hi_pri);

  return 0;
}
