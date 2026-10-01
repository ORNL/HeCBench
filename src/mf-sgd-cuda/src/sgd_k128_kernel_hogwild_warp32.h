__global__ void sgd_k128_kernel_hogwild_warp32_lrate(
    const mf_node *__restrict__ R,
    long long nnz,
    half *__restrict__ p,
    half *__restrict__ q,
    unsigned int *state,
    const float *__restrict__ dynamic_rate,
    long long u_seg,
    long long v_seg,
    int k,
    int num_iters,
    int current_iter,
    int update_count_per_block, 
    int update_count_this_block,
    int update_vector_size,
    float lambda_p,
    float lambda_q,
    int u_grid,
    int v_grid,
    int u_id,
    int v_id)
{
  //persistant thread
  for(int ite = current_iter; ite < current_iter + num_iters; ite ++)
  {
    float tmp_lrate = __ldg(&dynamic_rate[ite]);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {

      int lane_id = threadIdx.x%32;
      int local_wid = threadIdx.x/32;
      int wid = 4*blockIdx.x + local_wid;  

      long long start_id = 0;
      if(lane_id == 0)
      {
        long long origin = (long long)(LCG_random(state+wid)*nnz);
        start_id = origin%nnz;
      }
      start_id = __shfl_sync(0xffffffff, start_id, 0);

      for(int i = 0;i < update_vector_size;i++)
      {
        int offset = (start_id + i)%nnz;

        float r = __ldg(&R[offset].rate);
        int u = __ldg(&R[offset].u);
        int v = __ldg(&R[offset].v);

        //read the p & q into register file.
        int base_p = u*k;
        int base_q = v*k;

        float tmp_p1 = __half2float(p[base_p + lane_id]);
        float tmp_q1 = __half2float(q[base_q + lane_id]);

        float tmp_p2 = __half2float(p[base_p + lane_id + 32]);
        float tmp_q2 = __half2float(q[base_q + lane_id + 32]);

        float tmp_p3 = __half2float(p[base_p + lane_id + 64]);
        float tmp_q3 = __half2float(q[base_q + lane_id + 64]);

        float tmp_p4 = __half2float(p[base_p + lane_id + 96]);
        float tmp_q4 = __half2float(q[base_q + lane_id + 96]);

        float tmp_product = tmp_p1*tmp_q1 + tmp_p2*tmp_q2 + tmp_p3*tmp_q3 + tmp_p4*tmp_q4;

        //get dot product.
        tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
        tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
        tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
        tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
        tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
        tmp_product = __shfl_sync(0xffffffff, tmp_product,0);

        float ruv = r - tmp_product;

        //update
        //only works for k=blockDim.x=128
        p[base_p + lane_id +  0] = __float2half(tmp_p1 + tmp_lrate*(ruv*tmp_q1 - lambda_p*tmp_p1));
        q[base_q + lane_id +  0] = __float2half(tmp_q1 + tmp_lrate*(ruv*tmp_p1 - lambda_q*tmp_q1));

        p[base_p + lane_id + 32] = __float2half(tmp_p2 + tmp_lrate*(ruv*tmp_q2 - lambda_p*tmp_p2));
        q[base_q + lane_id + 32] = __float2half(tmp_q2 + tmp_lrate*(ruv*tmp_p2 - lambda_q*tmp_q2));

        p[base_p + lane_id + 64] = __float2half(tmp_p3 + tmp_lrate*(ruv*tmp_q3 - lambda_p*tmp_p3));
        q[base_q + lane_id + 64] = __float2half(tmp_q3 + tmp_lrate*(ruv*tmp_p3 - lambda_q*tmp_q3));

        p[base_p + lane_id + 96] = __float2half(tmp_p4 + tmp_lrate*(ruv*tmp_q4 - lambda_p*tmp_p4));
        q[base_q + lane_id + 96] = __float2half(tmp_q4 + tmp_lrate*(ruv*tmp_p4 - lambda_q*tmp_q4));
      }    
    }
  }
}



