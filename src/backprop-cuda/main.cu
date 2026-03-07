#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#include "backprop.h"

// cuda kernels
#include "bpnn_layerforward.h"
#include "bpnn_adjust_weights.h"

#include "reference.h"

double get_time() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
  setup(argc, argv);
  return 0;
}



int bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   

  float *input_weights_one_dim;
  float *input_weights_one_dim_r;
  float *input_weights_prev_one_dim;
  float * partial_sum;
  float sum;

  // Warning: the number of blocks must be less than the maximum grid dimension
  unsigned int num_blocks = in / BLOCK_SIZE;

  input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_one_dim_r = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
  partial_sum = (float *) malloc(num_blocks * WIDTH * sizeof(float));

  // this preprocessing stage is temporarily added to correct the bug of wrong memcopy using two-dimensional net->inputweights
  // todo: fix mem allocation
  int m = 0;
  for (int k = 0; k <= in; k++) {
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_one_dim_r[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
      m++;
    }
  }

  printf("Performing device offload\n");

  double offload_start = get_time();

  float* d_input;
  float *d_input_weights;
  float *d_hidden_partial_sum;
  float *d_hidden_delta;
  float *d_input_prev_weights;

  cudaMalloc((void**)&d_input, sizeof(float)*(in+1));
  cudaMalloc((void**)&d_input_weights, sizeof(float)*(in+1)*(hid+1));
  cudaMalloc((void**)&d_hidden_partial_sum, sizeof(float)*num_blocks*WIDTH);

  cudaMemcpy(d_input, net->input_units, sizeof(float)*(in+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_weights, input_weights_one_dim, sizeof(float)*(in+1)*(hid+1), cudaMemcpyHostToDevice);

  dim3 grid(1, num_blocks);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

  kernel_layerforward<<<grid, threads>>>(d_input, d_input_weights, d_hidden_partial_sum, hid);
  cudaMemcpy(partial_sum, d_hidden_partial_sum, sizeof(float)*num_blocks*WIDTH, cudaMemcpyDeviceToHost);

  for (int j = 1; j <= hid; j++) {
    sum = 0.f;
    for (unsigned int k = 0; k < num_blocks; k++) {  
      sum += partial_sum[k * hid + j-1] ;
    }
#ifdef DEBUG
    printf("j=%d sum=%f\n", j,sum);
#endif
    sum += net->input_weights[0][j];
    net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

  // input_weights has been written in the first kernel, so it needs to be restored.
  cudaMemcpy(d_input_weights, input_weights_one_dim, sizeof(float)*(in+1)*(hid+1), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_hidden_delta, sizeof(float)*(hid+1));
  cudaMalloc((void**)&d_input_prev_weights, sizeof(float)*(in+1)*(hid+1));
  cudaMemcpy(d_hidden_delta, net->hidden_delta, sizeof(float)*(hid+1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_prev_weights, input_weights_prev_one_dim, sizeof(float)*(in+1)*(hid+1), cudaMemcpyHostToDevice);
  kernel_adjust_weights<<<grid, threads>>>(d_input, d_input_weights, d_hidden_delta, d_input_prev_weights, hid);
  cudaMemcpy(input_weights_one_dim, d_input_weights, sizeof(float)*(in+1)*(hid+1), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_input_weights);
  cudaFree(d_hidden_partial_sum);
  cudaFree(d_hidden_delta);
  cudaFree(d_input_prev_weights);

  double offload_end = get_time();
  printf("Device offloading time = %lf(s)\n", offload_end - offload_start);

  reference (in, hid, out, net, 
             input_weights_one_dim_r,
             input_weights_prev_one_dim,
             partial_sum); 

  bool ok = true;
  for (int i = 0; i < (in+1)*(hid+1); i++) {
    if (fabsf(input_weights_one_dim[i] - input_weights_one_dim_r[i]) >= 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

#ifdef OUTPUT
  for (int i = 0; i < (in+1); i++) 
    printf("i=%d input_units=%f\n", i,net->input_units[i]);
  for (int i = 0; i < (in+1)*(hid+1); i++) 
    printf("i=%d input_weights=%f\n", i,input_weights_one_dim[i]);
#endif
  free(input_weights_prev_one_dim);
  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_one_dim_r);

  if (!ok) return 1;
  return 0;
}
