/*
   Kernels for softmax forward pass.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "common.h"
#include "reference.h"

// online softmax paper: http://arxiv.org/abs/1805.02867
// online softmax reduces loops from 3 to 2
// which is done by calculating sumval and maxval in one loop

void softmax_forward_baseline_kernel(int grid_size, int block_size, float* out, const float* inp, int N, int C) {
  #pragma omp target teams distribute num_teams(grid_size)
  for (int row = 0; row < N; row++) {
    const float* x = inp + row * C;
    float* const y = out + row * C;
    float maxval = -INFINITY, sumval = 0.0f;
    #pragma omp parallel for reduction(max:maxval) num_threads(block_size)
    for (int i = 0; i < C; i++) {
      maxval = fmaxf(x[i], maxval);
    }
    #pragma omp parallel for reduction(+:sumval) num_threads(block_size)
    for (int i = 0; i < C; i++) {
      sumval += expf(x[i] - maxval);
    }
    #pragma omp parallel for num_threads(block_size)
    for (int i = 0; i < C; i++) {
      y[i] = expf(x[i] - maxval) / sumval;
    }
  }
}

// struct for the reduction operation, matching the CUDA SumMax
struct SumMax {
  float maxval;
  float sum;
};

// -inf - -inf is NaN, and 0 * NaN stays NaN, so a later finite value cannot
// recover the sum. Use 0 as the exp shift while the running max is still -inf.
#pragma omp declare target
inline float softmax_exp_shift(float maxval) {
  return (maxval == -INFINITY) ? 0.0f : maxval;
}

inline SumMax reduce_sum_max_op(SumMax a, SumMax b) {
  bool a_bigger = (a.maxval > b.maxval);
  SumMax bigger_m = a_bigger ? a : b;
  SumMax smaller_m = a_bigger ? b : a;
  SumMax res;
  res.maxval = bigger_m.maxval;
  float shift = softmax_exp_shift(res.maxval);
  res.sum = bigger_m.sum + smaller_m.sum * expf(smaller_m.maxval - shift);
  return res;
}
#pragma omp end declare target

// CUDA softmax_forward_online_kernel: each thread scans a stride of the row,
// then cg::reduce merges the {max, sum} pairs with reduce_sum_max_op.
// nvc++ rejects declare reduction for a user-defined type, so the tree is written out.
void softmax_forward_online_kernel(int grid_size, int block_size, float* out, const float* inp, int N, int C) {
  #pragma omp target teams num_teams(grid_size) thread_limit(block_size)
  {
    SumMax smem[1024];
    #pragma omp parallel
    {
      int row = omp_get_team_num();
      if (row < N) {
        const float* x = inp + row * C;
        float* const y = out + row * C;

        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int pow2 = 1;
        while (pow2 * 2 <= nthreads) pow2 *= 2;

        SumMax sm_partial;
        sm_partial.maxval = -INFINITY;
        sm_partial.sum = 0.0f;
        for (int i = tid; i < C; i += nthreads) {
          SumMax v;
          v.maxval = x[i];
          v.sum = 1.0f;
          sm_partial = reduce_sum_max_op(sm_partial, v);
        }
        smem[tid] = sm_partial;
        #pragma omp barrier

        if (tid >= pow2) smem[tid - pow2] = reduce_sum_max_op(smem[tid - pow2], smem[tid]);
        #pragma omp barrier

        for (int stride = pow2 / 2; stride > 0; stride /= 2) {
          if (tid < stride) smem[tid] = reduce_sum_max_op(smem[tid], smem[tid + stride]);
          #pragma omp barrier
        }

        SumMax sm_total = smem[0];
        #pragma omp barrier

        for (int i = tid; i < C; i += nthreads) {
          y[i] = expf(x[i] - sm_total.maxval) / sm_total.sum;
        }
      }
    }
  }
}

// CUDA softmax_forward_online_kernel3: shared-memory tree reduction
void softmax_forward_online3_kernel(int grid_size, int block_size, float* out, const float* inp, int N, int C) {

  #pragma omp target teams num_teams(grid_size) thread_limit(block_size) 
  {
    float smem[1024];
    #pragma omp parallel
    {
      int row = omp_get_team_num();
      if (row < N) {
        const float* x = inp + row * C;
        float* const y = out + row * C;

        float maxval = -INFINITY, sumval = 0.0f;

        // thread_limit is an upper bound; the team may have fewer threads
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int pow2 = 1;
        while (pow2 * 2 <= nthreads) pow2 *= 2;

        for (int i = tid; i < C; i += nthreads) {
          float v = x[i];
          float bigger = fmaxf(maxval, v);
          float shift = softmax_exp_shift(bigger);
          if (v > maxval) {
            sumval *= expf(maxval - shift);
            maxval = v;
          }
          sumval += expf(v - shift);
        }
        smem[tid] = maxval;
        #pragma omp barrier

        if (tid >= pow2) smem[tid - pow2] = fmaxf(smem[tid - pow2], smem[tid]);
        #pragma omp barrier

        for (int stride = pow2 / 2; stride > 0; stride /= 2) {
          if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
          #pragma omp barrier
        }

        float global_maxval = smem[0];
        #pragma omp barrier

        smem[tid] = sumval * expf(maxval - global_maxval);
        #pragma omp barrier

        if (tid >= pow2) smem[tid - pow2] += smem[tid];
        #pragma omp barrier

        for (int stride = pow2 / 2; stride > 0; stride /= 2) {
           if (tid < stride) smem[tid] += smem[tid + stride];
           #pragma omp barrier
        }

        float global_sum = smem[0];
        #pragma omp barrier

        for (int i = tid; i < C; i += nthreads) {
          y[i] = expf(x[i] - global_maxval) / global_sum;
        }
      }
    }
  }
}


void softmax_forward_baseline(float* out, const float* inp, int N, int C, int block_size) {
  const int grid_size = N;
  softmax_forward_baseline_kernel(grid_size, block_size, out, inp, N, C);
}

void softmax_forward_online(float* out, const float* inp, int N, int C, int block_size) {
  const int grid_size = N;
  softmax_forward_online_kernel(grid_size, block_size, out, inp, N, C);
}

void softmax_forward_online3(float* out, const float* inp, int N, int C, int block_size) {
  const int grid_size = N;
  softmax_forward_online3_kernel(grid_size, block_size, out, inp, N, C);
}

// kernel version dispatch
void softmax_forward(int kernel_num, float* out, const float* inp, int N, int C,
                     const int block_size) {
  switch (kernel_num) {
    case 1:
      softmax_forward_baseline(out, inp, N, C, block_size);
      break;
    case 2:
      softmax_forward_online(out, inp, N, C, block_size);
      break;
    case 4:
      softmax_forward_online3(out, inp, N, C, block_size);
      break;
    default:
      printf("Invalid kernel number\n");
      exit(1);
  }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  srand(0);

  int B = 8;
  int T = 1024;
  int V = 50257;

  // create host memory of random numbers
  float* out = (float*)malloc(B * T * V * sizeof(float));
  float* d_out = (float*)malloc(B * T * V * sizeof(float));
  float* inp = make_random_float(B * T * V);

  // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
  // and the tests are not actually meaningful.
  const int* outliers = make_random_int(B * T * 3, V);
  for(int k = 0; k < 3; ++k) {
    for(int j = 0; j < B * T; ++j) {
      inp[j * V + outliers[j*3 + k]] *= 20;
    }
  }

  // read kernel_num from command line
  int kernel_num = 2;
  if (argc > 1) {
    kernel_num = atoi(argv[1]);
  }
  if (kernel_num > 1)
    printf("Using kernel online %d\n", kernel_num);
  else
    printf("Using kernel baseline %d\n", kernel_num);

  softmax_forward_cpu(out, inp, B * T, V);
  {
    float max_el = -INFINITY;
    for(int i = 0; i <  B * T * V; ++i) {
      max_el = fmaxf(max_el, out[i]);
    }
    assert(max_el > 1e-4);
    printf("Largest output is: %f\n", max_el);
  }

  #pragma omp target data map(to: inp[0:B * T * V]) map(alloc: d_out[0:B * T * V])
  {
    // first check the correctness of the kernel
    for (int j = 64; j <= 1024; j = j * 2) {
      int block_size = j;
      printf("Checking block size %d.\n", block_size);
      softmax_forward(kernel_num, d_out, inp, B * T, V, block_size);
      validate_result(d_out, out, "out", B * T * V, 1e-4f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 64; j <= 1024; j = j * 2) {
      int block_size = j;
      int repeat_times = 100;
      float elapsed_time = benchmark_kernel(repeat_times, softmax_forward,
                                            kernel_num, d_out, inp, B * T, V, block_size);
      printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (B*T));
    }
  }

  // free memory
  free(out);
  free(d_out);
  free(inp);
  free((void*)outliers);

  return 0;
}
