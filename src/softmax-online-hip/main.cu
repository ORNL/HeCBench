/*
   Kernels for softmax forward pass.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <hipcub/hipcub.hpp>
#include "common.h"
#include "reference.h"

// online softmax paper: http://arxiv.org/abs/1805.02867
// online softmax reduces loops from 3 to 2
// which is done by calculating sumval and maxval in one loop

/* struct for the reduction operation, guarantees 8-byte alignment
struct __align__(8) SumMax
{
    float maxval;
    float sum;
};

// forceinline helps avoid function call overhead
__device__ __forceinline__ SumMax reduce_sum_max_op(SumMax a, SumMax b) {
    bool a_bigger = (a.maxval > b.maxval);
    SumMax bigger_m = a_bigger ? a : b;
    SumMax smaller_m = a_bigger ? b : a;
    SumMax res;
    res.maxval = bigger_m.maxval;
    res.sum = bigger_m.sum + smaller_m.sum * expf(smaller_m.maxval - bigger_m.maxval);
    return res;
}

__global__ void softmax_forward_online_kernel(float* out, const float* inp, int N, int C) {
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  int row = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (row >= N) {
    return;
  }

  // one row of inp, i.e. inp[row, :] of shape (C,)
  const float* x = inp + row * C;
  float* const y = out + row * C;

  // base case for the reduction
  SumMax sm_partial;
  sm_partial.maxval = -INFINITY;
  sm_partial.sum = 0.0f;

  // first, thread coarsening by directly accessing global memory in series
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    sm_partial = reduce_sum_max_op(sm_partial, { x[i], 1.0f });
  }

  // second, the reduction
  SumMax sm_total = cg::reduce(warp, sm_partial, reduce_sum_max_op);

  // divide the whole row by the sum
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    y[i] = expf(x[i] - sm_total.maxval) / sm_total.sum;
  }
}
*/

// -inf - -inf is NaN, and 0 * NaN stays NaN, so a later finite value cannot
// recover the sum. Use 0 as the exp shift while the running max is still -inf.
__device__ __forceinline__ float softmax_exp_shift(float maxval) {
  return (maxval == -INFINITY) ? 0.0f : maxval;
}

template <unsigned int WarpSize>
__global__ void softmax_forward_online_kernel2(float* out, const float* inp, int N, int C) {
  int tid = threadIdx.x;
  if (tid >= C) return;

  int warpId = tid / WarpSize;
  const int warpsPerBlock = blockDim.x / WarpSize;
  int row = blockIdx.x * warpsPerBlock + warpId;

  if (row >= N) return;

  int laneId = tid % WarpSize;
  const float* x = inp + row * C;
  float* const y = out + row * C;

  // merge calculating maxval and sumval in one loop
  // which is an arithmetic improvment from online softmax over normal softmax
  float maxval = -INFINITY, sumval = 0.0f, bigger;
  for (int i = laneId; i < C; i += WarpSize) {
    // when updating the maxval, dynamically updates the previous sumval by
    // multiplying e^{previous_maxval - current_maxval}
    bigger = fmaxf(maxval, x[i]);
    float shift = softmax_exp_shift(bigger);
    sumval = sumval * expf(maxval - shift) + expf(x[i] - shift);
    maxval = bigger;
  }

  // use warp functions instead of cooperative groups for better readibility
  // calculate the warp wised maxval and sumval
  float offsetMaxval, offsetSumval;
  for (int offset = WarpSize / 2; offset > 0; offset >>= 1) {
    offsetMaxval = __shfl_down(maxval, offset);
    offsetSumval = __shfl_down(sumval, offset);
    if (offsetMaxval > maxval) {
      sumval *= expf(maxval - offsetMaxval);
      maxval = offsetMaxval;
    } else {
      offsetSumval *= expf(offsetMaxval - softmax_exp_shift(maxval));
    }
    sumval += offsetSumval;
  }

  // sync the warp wised maxval and sumval
  // which are also the maxval and sumval of one row in C
  maxval = __shfl(maxval, 0);
  sumval = __shfl(sumval, 0);

  for (int i = laneId; i < C; i += WarpSize) {
    y[i] = expf(x[i] - maxval) / sumval;
  }
}

__global__ void softmax_forward_online_kernel3(float* __restrict__ out, const float* __restrict__ inp, int N, int C) {
  __shared__ float smem[1024];

  int row = blockIdx.x;
  if (row >= N) return;

  const float* x = inp + row * C;
  float* y = out + row * C;
  float maxval = -INFINITY;
  float sumval = 0.0f;

  int tid = threadIdx.x;
  for (int i = tid; i < C; i += blockDim.x) {
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
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
          smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
      }
      __syncthreads();
  }

  float row_max = smem[0];
  __syncthreads();

  smem[tid] = sumval * expf(maxval - row_max);
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
          smem[tid] += smem[tid + stride];
      }
      __syncthreads();
  }
  float row_sum = smem[0];
  __syncthreads();

  for (int i = tid; i < C; i += blockDim.x) {
      y[i] = expf(x[i] - row_max) / row_sum;
  }
}

// Baseline softmax
template <unsigned int WarpSize>
__global__ void softmax_forward_baseline_kernel(float* out, const float* inp, int N, int C) {
  int tid = threadIdx.x;
  if (tid >= C) return;

  int warpId = tid / WarpSize;
  const int warpsPerBlock = blockDim.x / WarpSize;
  int row = blockIdx.x * warpsPerBlock + warpId;

  if (row >= N) return;

  int laneId = tid % WarpSize;
  const float* x = inp + row * C;
  float* const y = out + row * C;

  using WarpReduce = hipcub::WarpReduce<float>;
  __shared__ typename WarpReduce::TempStorage temp[1024/WarpSize];

  float maxval = -INFINITY;
  for (int i = laneId; i < C; i += WarpSize) {
    maxval = max(x[i], maxval);
  }
  maxval = WarpReduce(temp[warpId]).Reduce(maxval, hipcub::Max());

  maxval = __shfl(maxval, 0);
  
  float sumval = 0;
  for (int i = laneId; i < C; i += WarpSize) {
    sumval += expf(x[i] - maxval);
  }
  sumval = WarpReduce(temp[warpId]).Sum(sumval);

  sumval = __shfl(sumval, 0);

  for (int i = laneId; i < C; i += WarpSize) {
    y[i] = expf(x[i] - maxval) / sumval;
  }
}

void softmax_forward_baseline(float* out, const float* inp, int N, int C,
                            int warp_size, int block_size) {
  const int grid_size = ceil_div(N * warp_size, block_size);
  if (warp_size == 64)
    softmax_forward_baseline_kernel<64><<<grid_size, block_size >>> (out, inp, N, C);
  else
    softmax_forward_baseline_kernel<32><<<grid_size, block_size >>> (out, inp, N, C);
  hipCheck(hipDeviceSynchronize());
}

/*
void softmax_forward_online(float* out, const float* inp, int N, int C,
                            int warp_size, int block_size) {
  const int grid_size = ceil_div(N * warp_size, block_size);
  softmax_forward_online_kernel <<<grid_size, block_size >>> (out, inp, N, C);
  hipCheck(hipDeviceSynchronize());
}
*/

void softmax_forward_online2(float* out, const float* inp, int N, int C,
                             int warp_size, int block_size) {
  const int grid_size = ceil_div(N * warp_size, block_size);
  if (warp_size == 64)
    softmax_forward_online_kernel2<64><<<grid_size, block_size >>> (out, inp, N, C);
  else
    softmax_forward_online_kernel2<32><<<grid_size, block_size >>> (out, inp, N, C);
  hipCheck(hipDeviceSynchronize());
}

void softmax_forward_online3(float* out, const float* inp, int N, int C,
                             int warp_size, int block_size) {
  const int grid_size = N;
  softmax_forward_online_kernel3 <<<grid_size, block_size >>> (out, inp, N, C);
  hipCheck(hipDeviceSynchronize());
}

// kernel version dispatch
void softmax_forward(int kernel_num, float* out, const float* inp, int N, int C,
                     const int warp_size, const int block_size) {
  switch (kernel_num) {
    case 1:
      softmax_forward_baseline(out, inp, N, C, warp_size, block_size);
      break;
    case 2:
      //softmax_forward_online(out, inp, N, C, warp_size, block_size);
      printf("kernel not supported\n");
      break;
    case 3:
      softmax_forward_online2(out, inp, N, C, warp_size, block_size);
      break;
    case 4:
      softmax_forward_online3(out, inp, N, C, warp_size, block_size);
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
  float* inp = make_random_float(B * T * V);

  // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
  // and the tests are not actually meaningful.
  const int* outliers = make_random_int(B * T * 3, V);
  for(int k = 0; k < 3; ++k) {
    for(int j = 0; j < B * T; ++j) {
      inp[j * V + outliers[j*3 + k]] *= 20;
    }
  }

  // move to GPU
  float* d_out;
  float* d_inp;
  hipCheck(hipMalloc((void**)&d_out, B * T * V * sizeof(float)));
  hipCheck(hipMalloc((void**)&d_inp, B * T * V * sizeof(float)));
  hipCheck(hipMemcpy(d_inp, inp, B * T * V * sizeof(float), hipMemcpyHostToDevice));

  // read kernel_num from command line
  int kernel_num = 2;
  if (argc > 1) {
    kernel_num = atoi(argv[1]);
  }
  if (kernel_num > 1)
    printf("Using kernel online %d\n", kernel_num);
  else
    printf("Using kernel baseline %d\n", kernel_num);


  // query the warp size
  int warp_size;
  hipCheck(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  softmax_forward_cpu(out, inp, B * T, V);
  {
    float max_el = -INFINITY;
    for(int i = 0; i <  B * T * V; ++i) {
      max_el = fmaxf(max_el, out[i]);
    }
    assert(max_el > 1e-4);
    printf("Largest output is: %f\n", max_el);
  }

  // first check the correctness of the kernel
  for (int j = warp_size; j <= 1024; j = j * 2) {
    int block_size = j;
    printf("Checking block size %d.\n", block_size);
    softmax_forward(kernel_num, d_out, d_inp, B * T, V, warp_size, block_size);
    validate_result(d_out, out, "out", B * T * V, 1e-4f);
  }

  printf("All results match. Starting benchmarks.\n\n");

  // time the kernel at different block sizes
  for (int j = warp_size; j <= 1024; j = j * 2) {
    int block_size = j;
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, softmax_forward,
                                          kernel_num, d_out, d_inp, B * T, V,
                                          warp_size, block_size);
    printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (B*T));
  }

  // free memory
  free(out);
  free(inp);
  free((void*)outliers);
  hipCheck(hipFree(d_out));
  hipCheck(hipFree(d_inp));

  return 0;
}
