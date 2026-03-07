#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "reference.h"

__global__
void mha (
   const float *__restrict__ q, 
   const float *__restrict__ k, 
   const float *__restrict__ v, 
   const int beam_size, 
   const int n_steps, 
   const int qk_col, 
   const int v_col, 
   const int nhead, 
   const float scale,
   const int THRESHOLD,
   float *__restrict__ dst)
{
  /* 
     Each block processes one head from one candidate.

     dim_per_head is the size of partition processed by each head.

     candidate_id is the index of candidate processed by this block. 
     We have beam_size candidates in total.

     head_id is the index of head processed by this block.

   */
  int dim_per_head = qk_col / nhead;
  int candidate_id = blockIdx.x / nhead;
  int head_id = blockIdx.x % nhead;

  typedef hipcub::BlockReduce<float, 256> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  /*
     sq is the query vector shared by all threads inside the same block.

     The size of sq should be dim_per_head.

     Each block only load the a part of the query vector that belongs to the corresponding candidate.
   */
  extern __shared__ float buffer[];
  float *sq = buffer;
  float *logits = buffer + dim_per_head;


  // pos is the start position of the corresponding query matrix prococessed by this block.
  int pos = candidate_id * qk_col + head_id * dim_per_head + threadIdx.x;
  if(threadIdx.x < dim_per_head) sq[threadIdx.x] = q[pos];
  __syncthreads();

  // calculate the correlation between the query and key QK^T/sqrt(d_k)

  float summ = 0.f;
  if(threadIdx.x < n_steps)
  {   
    const float *k2 = k + candidate_id * qk_col * n_steps + head_id * dim_per_head + threadIdx.x * qk_col;
    for (int i = 0; i < dim_per_head; i++)
      summ += sq[i] * k2[i];
    summ *= scale;
  }   

  // calculate the softmax value of the first step softmax(QK^T/sqrt(d_k)) using warp shuffle.

  __shared__ float s_max_val;
  __shared__ float s_sum;

  float local_i = threadIdx.x < n_steps ? summ : -1e-20f;
  float local_o;

  float max_val = BlockReduce(temp_storage).Reduce(local_i, hipcub::Max());

  if(threadIdx.x == 0)
    s_max_val = max_val;
  __syncthreads();

  local_i -= s_max_val;

  if(local_i < -THRESHOLD) local_i = -THRESHOLD;

  local_o = expf(local_i);

  float val = (threadIdx.x < n_steps) ? local_o : 0.f;
  val = BlockReduce(temp_storage).Sum(val);
  if(threadIdx.x == 0) s_sum = val;
  __syncthreads();

  if(threadIdx.x < n_steps) logits[threadIdx.x] = local_o / s_sum;
  __syncthreads();

  // calculate the weighted sum on value matrix V softmax(QK^T/sqrt(d_k))V 
  summ = 0.f;
  if(threadIdx.x < dim_per_head)
  {
    int tid = candidate_id * v_col * n_steps + head_id * dim_per_head + threadIdx.x;
    for(int i = 0; i < n_steps; ++i)
      summ += logits[i] * v[tid + i * v_col];
    dst[candidate_id * v_col + head_id * dim_per_head + threadIdx.x] = summ;
  }
}

/*
q: Query matrix of dimension beam_size * qk_col
k: Key matrix of dimension beam_size * (n_steps * qk_col)
v: Value matrix of dimension beam_size * (n_steps * v_col)
beamsize: Dimension used in beam size, also called the number of candidates
n_steps: The number of words that have already been decoded
qk_col: Dimension of the query feature
v_col: Dimension of the value feature
nhead: The number of heads
scaler: Pre-computed scaler 
THRESHOLD: Customer-defined value for soft-max maximum value calculation
dst: Output. The attention value of this query over all decoded word keys
 */

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  /*
     Each step, we maintain beamsize candidates for the beam search.

     We have nhead heads, which contains dim_feature/nhead values.

     Currently, we have already decoded n_steps words.

   */
  const int beamsize = 4;
  const int nhead = 16;
  const int dim_feature = nhead * 256;
  const int n_steps = 9;

  // Calculate sqrt(d_k) 
  const float scaler = sqrtf(nhead * 1.f / dim_feature);

  //qk_col can be different with v_col.
  const int qk_col = dim_feature;
  const int v_col = dim_feature;
  const int THRESHOLD = 64;

  const int q_size = beamsize * dim_feature;
  const int q_size_bytes = sizeof(float) * q_size;

  const int k_size = beamsize * dim_feature * n_steps;
  const int k_size_bytes = sizeof(float) * k_size;

  const int v_size = beamsize * dim_feature * n_steps;
  const int v_size_bytes = sizeof(float) * v_size;

  float *dq, *dk, *dv, *dst;
  hipMalloc((void**)&dq, q_size_bytes);
  hipMalloc((void**)&dk, k_size_bytes);
  hipMalloc((void**)&dv, v_size_bytes);
  hipMalloc((void**)&dst, q_size_bytes);

  float *hq = (float*)malloc(q_size_bytes);
  float *hk = (float*)malloc(k_size_bytes);
  float *hv = (float*)malloc(v_size_bytes);
  float *h_dst = (float*)malloc(q_size_bytes);
  float *r_dst = (float*)malloc(q_size_bytes);

  // Initialize query, key and value matrices
  srand(123);
  for(int i = 0; i < q_size; ++i)
    hq[i] = rand() / (float)RAND_MAX;

  for(int i = 0; i < k_size; ++i)
    hk[i] = rand() / (float)RAND_MAX;

  for(int i = 0; i < v_size; ++i)
    hv[i] = rand() / (float)RAND_MAX;

  hipMemcpy(dq, hq, q_size_bytes, hipMemcpyHostToDevice);
  hipMemcpy(dk, hk, k_size_bytes, hipMemcpyHostToDevice);
  hipMemcpy(dv, hv, v_size_bytes, hipMemcpyHostToDevice);

  dim3 grid(nhead * beamsize);
  dim3 block(qk_col / nhead);

  const int shared_size = sizeof(float) * ((qk_col / nhead) + n_steps);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    hipLaunchKernelGGL(mha, grid, block, shared_size, 0 , dq, dk, dv,
      beamsize, n_steps, qk_col, v_col, nhead, scaler, THRESHOLD, dst);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  hipMemcpy(h_dst, dst, q_size_bytes, hipMemcpyDeviceToHost);

  hipFree(dq);
  hipFree(dk);
  hipFree(dv);
  hipFree(dst);

  mha_reference(hq, hk, hv, beamsize, n_steps, qk_col, v_col, nhead, scaler, THRESHOLD, r_dst);

  bool ok = true;
  for (int i = 0; i < beamsize; i++) {
    for (int j = 0; j < dim_feature; j++) {
      if (fabsf(h_dst[i*dim_feature+j] - r_dst[i*dim_feature+j]) > 1e-3f) {
        ok = false;
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");    
  
  free(hq);
  free(hk);
  free(hv);
  free(h_dst);
  free(r_dst);

  if (!ok) return 1;
  return 0;
}
