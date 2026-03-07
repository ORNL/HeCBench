#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "reference.h"

#define GPU_NUM_THREADS 256

float k_sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * (tgt - (lgt >= 0.f)) - sycl::log(1.f + sycl::exp(lgt - 2.f * lgt * (lgt >= 0.f)));
}

float k_sigmoid_partition(float lgt) {
  return lgt * (lgt >= 0.f) + sycl::log(1.f + sycl::exp(lgt - 2.f * lgt * (lgt >= 0.f)));
}

float k_sigmoid_xent_forward_with_log_d_trick(float lgt, float tgt) {
  return (2.f * tgt - 1.f) * (lgt - k_sigmoid_partition(lgt));
}

float k_unjoined_sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * tgt + (tgt - 1.f) * lgt * (lgt >= 0.f) -
      (1.f - tgt) * sycl::log(1.f + sycl::exp(lgt - 2.f * lgt * (lgt >= 0.f)));
}

void SigmoidCrossEntropyWithLogitsKernel(
  sycl::nd_item<1> &item,
  const int inner_size,
  const bool log_D_trick,
  const bool unjoined_lr_loss,
  const float* logits_ptr,
  const float* targets_ptr,
        float* out_ptr)
{
  int i = item.get_group(0);
  int last_idx = (i + 1) * inner_size;
  float value = 0;
  for (int in_idx = i * inner_size + item.get_local_id(0);
           in_idx < last_idx; in_idx += item.get_local_range(0)) {
    float lgt = logits_ptr[in_idx];
    float tgt = targets_ptr[in_idx];
    if (unjoined_lr_loss) {
      value += k_unjoined_sigmoid_xent_forward(lgt, tgt);
    } else {
      value += log_D_trick ?
               k_sigmoid_xent_forward_with_log_d_trick(lgt, tgt) :
               k_sigmoid_xent_forward(lgt, tgt);
    }
  }

  float sum = reduce_over_group(item.get_group(), value, std::plus<>());
  if (item.get_local_id(0) == 0) {
    out_ptr[i] = -sum / inner_size;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <outer size> <inner_size> <repeat>\n", argv[0]);
    return 1;
  }

  const int outer_size = atoi(argv[1]);
  const int inner_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  int input_size = (outer_size + 1) * inner_size;
  int input_size_bytes = input_size * sizeof(float);

  int output_size = outer_size;
  int output_size_bytes = output_size * sizeof(float);

  std::default_random_engine generator (123);
  std::normal_distribution<float> distribution(0, 1);

  float *h_logits = (float*) malloc (input_size_bytes);
  float *h_targets = (float*) malloc (input_size_bytes);
  float *h_out = (float*) malloc (output_size_bytes);
  float *r_out = (float*) malloc (output_size_bytes);

  for (int i = 0; i < input_size; i++) {
    h_logits[i] = distribution(generator);
    h_targets[i] = distribution(generator) + 1.f;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_logits, *d_targets, *d_out;
  d_logits = sycl::malloc_device<float>(input_size, q);

  q.memcpy(d_logits, h_logits, input_size_bytes);

  d_targets = sycl::malloc_device<float>(input_size, q);
  q.memcpy(d_targets, h_targets, input_size_bytes);

  d_out = sycl::malloc_device<float>(output_size, q);

  bool ok = true;

  sycl::range<1> gws (outer_size * GPU_NUM_THREADS);
  sycl::range<1> lws (GPU_NUM_THREADS);

  for (int unjoined_lr_loss = 0; unjoined_lr_loss <= 1; unjoined_lr_loss++) {

    int logD = (unjoined_lr_loss == 0) ? 1 : 0;

    for (int logD_trick = 0; logD_trick <= logD; logD_trick++) {

      q.wait();
      auto start = std::chrono::steady_clock::now();

      for (int i = 0; i < repeat; i++) {
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
            SigmoidCrossEntropyWithLogitsKernel(
              item,
              inner_size,
              logD_trick,
              unjoined_lr_loss,
              d_logits,
              d_targets,
              d_out);
          });
        });
      }

      q.wait();
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of SigmoidCrossEntropyWithLogits kernel: %f (us)\n",
             (time * 1e-3f) / repeat);

      q.memcpy(h_out, d_out, output_size_bytes).wait();

      reference (outer_size, inner_size, logD_trick, unjoined_lr_loss, h_logits, h_targets, r_out);
      for (int i = 0; i < output_size; i++) {
        if (fabsf(r_out[i] - h_out[i]) > 1e-3f) {
          ok = false;
          break;
        }
      }
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_targets, q);
  sycl::free(d_logits, q);
  sycl::free(d_out, q);

  free(h_targets);
  free(h_logits);
  free(h_out);
  free(r_out);

  if (!ok) return 1;
  return 0;
}
