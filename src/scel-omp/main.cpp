#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "reference.h"

#define GPU_NUM_THREADS 256

void SigmoidCrossEntropyWithLogitsKernel(
  const int outer_size,
  const int inner_size,
  const bool log_D_trick,
  const bool unjoined_lr_loss,
  const float* logits_ptr,
  const float* targets_ptr,
        float* out_ptr)
{
  #pragma omp target teams distribute num_teams(outer_size)
  for (int i = 0; i < outer_size; i++) {
    float value = 0;
    #pragma omp parallel for reduction(+:value) num_threads(GPU_NUM_THREADS)
    for (int in_idx = i * inner_size;
             in_idx < (i+1) * inner_size; in_idx++) {
      float lgt = logits_ptr[in_idx];
      float tgt = targets_ptr[in_idx];
      if (unjoined_lr_loss) {
        value += unjoined_sigmoid_xent_forward(lgt, tgt);
      } else {
        value += log_D_trick ?
                 sigmoid_xent_forward_with_log_d_trick(lgt, tgt) :
                 sigmoid_xent_forward(lgt, tgt);
      }
    }
    out_ptr[i] = -value / inner_size;
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

  bool ok = true;

  #pragma omp target data map(to: h_logits[0:input_size],\
                                  h_targets[0:input_size]) \
                          map(alloc: h_out[0:output_size]) 
  {
    for (int unjoined_lr_loss = 0; unjoined_lr_loss <= 1; unjoined_lr_loss++) {

      int logD = (unjoined_lr_loss == 0) ? 1 : 0;

      for (int logD_trick = 0; logD_trick <= logD; logD_trick++) {

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < repeat; i++) {
          SigmoidCrossEntropyWithLogitsKernel(
            outer_size,
            inner_size,
            logD_trick,
            unjoined_lr_loss,
            h_logits,
            h_targets,
            h_out);
        }

        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average execution time of SigmoidCrossEntropyWithLogits kernel: %f (us)\n",
               (time * 1e-3f) / repeat);

        #pragma omp target update from (h_out[0:output_size]) 

        reference (outer_size, inner_size, logD_trick, unjoined_lr_loss, h_logits, h_targets, r_out);
        for (int i = 0; i < output_size; i++) {
          if (fabsf(r_out[i] - h_out[i]) > 1e-3f) {
            ok = false;
            break;
          }
        }
      }
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(h_targets);
  free(h_logits);
  free(h_out);
  free(r_out);

  return 0;
}
