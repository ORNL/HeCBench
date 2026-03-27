#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <sycl/sycl.hpp>
#include "kernels.h"
#include "reference.h"

//----------------------------------------------------------------
//   high priority  - forward + backward (blocks training loop)
//   low  priority  - data prefetch for next batch (background)
//----------------------------------------------------------------
void use_case_ml_training(int batches) try {
  const int N       = 1 << 24;
  const int threads = 256;
  const int blocks  = (N + threads - 1) / threads;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // high and low priority streams
  auto device = q.get_device();
  sycl::queue q_train(device, sycl::property_list{sycl::property::queue::in_order(),
                      sycl::ext::oneapi::property::queue::priority_high()});
  sycl::queue q_prefetch(device, sycl::property_list{sycl::property::queue::in_order(),
                         sycl::ext::oneapi::property::queue::priority_low()});


  std::vector<float> prefetch_buf (N, 0);
  std::vector<float> weights (N, 1);
  std::vector<float> weights_ref (N, 1); // constant
  std::vector<float> grad_in (N, 1);
  std::vector<float> grad_in_ref (N, 1); // constant
  std::vector<float> act (N);
  std::vector<float> grad_out (N);
  std::vector<float> grad_out_dev (N); // from device

  float *d_act, *d_grad_in, *d_grad_out, *d_prefetch_buf, *d_weights;
  d_act = sycl::malloc_device<float>(N, q);
  d_grad_in = sycl::malloc_device<float>(N, q);
  d_grad_out = sycl::malloc_device<float>(N, q);
  d_prefetch_buf = sycl::malloc_device<float>(N, q);
  d_weights = sycl::malloc_device<float>(N, q);
  q.memset(d_prefetch_buf, 0, N * sizeof(float));
  q.memcpy(d_grad_in, grad_in_ref.data(), N * sizeof(float));
  q.memcpy(d_weights, weights_ref.data(), N * sizeof(float));

  //------------------------------------------------------------
  // reference
  //------------------------------------------------------------
  for (int b = 0; b < batches; b++) {
    forward_pass(prefetch_buf.data(), act.data(), N, 1.0f);
    backward_pass(act.data(), grad_in.data(), grad_out.data(), N);
    sgd_update(weights.data(), grad_out.data(), N, 0.01f);
    data_prefetch(prefetch_buf.data(), N, b + 1);
  }

  sycl::range<1> gws (blocks * threads);
  sycl::range<1> lws (threads);
  
  //------------------------------------------------------------
  printf("Default stream (no priority)\n");
  //------------------------------------------------------------
  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int b = 0; b < batches; b++) {
    q.parallel_for(sycl::nd_range<1>(gws, lws),
                       [=](sycl::nd_item<1> item) {
                         forward_pass_kernel(d_prefetch_buf, d_act, N, 1.0f, item);
                       });
    q.parallel_for(sycl::nd_range<1>(gws, lws),
                       [=](sycl::nd_item<1> item) {
                         backward_pass_kernel(d_act, d_grad_in, d_grad_out, N, item);
                       });
    q.parallel_for(sycl::nd_range<1>(gws, lws),
                       [=](sycl::nd_item<1> item) {
                         sgd_update_kernel(d_weights, d_grad_out, N, 0.01f, item);
                       });
    q.parallel_for(sycl::nd_range<1>(gws, lws),
                       [=](sycl::nd_item<1> item) {
                         data_prefetch_kernel(d_prefetch_buf, N, b+1, item);
                       });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time_default = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.memcpy(grad_out_dev.data(), d_grad_out, N * sizeof(float)).wait();
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
  q.memset(d_prefetch_buf, 0, N * sizeof(float));
  q.memcpy(d_grad_in, grad_in_ref.data(), N * sizeof(float));
  q.memcpy(d_weights, weights_ref.data(), N * sizeof(float));

  q.wait();
  start = std::chrono::steady_clock::now();

  sycl::event fwd_done, fetch_done;

  for (int b = 0; b < batches; b++) {
    fwd_done = q_train.submit([&](sycl::handler &cgh) {
      cgh.depends_on(fetch_done); 
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        forward_pass_kernel(d_prefetch_buf, d_act, N, 1.0f, item);
      });
    });
    q_train.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      backward_pass_kernel(d_act, d_grad_in, d_grad_out, N, item);
    });
    q_train.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      sgd_update_kernel(d_weights, d_grad_out, N, 0.01f, item);
    });
    fetch_done = q_prefetch.submit([&](sycl::handler &cgh) {
      cgh.depends_on(fwd_done);
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        data_prefetch_kernel(d_prefetch_buf, N, b + 1, item);
      });
    });
  }
  q_train.wait(); // wait for q_train stream
  end = std::chrono::steady_clock::now();
  auto time_pri = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q_train.memcpy(grad_out_dev.data(), d_grad_out, N * sizeof(float)).wait();

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

  sycl::free(d_act, q);
  sycl::free(d_grad_in, q);
  sycl::free(d_grad_out, q);
  sycl::free(d_prefetch_buf, q);
  sycl::free(d_weights, q);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  use_case_ml_training(repeat);

  return 0;
}
