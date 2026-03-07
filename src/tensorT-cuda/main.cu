#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda.h>

#define TILE_SIZE 5900
#define NTHREADS 256

// 1,2,3,4,5,6 -> 2,3,4,6,1,5
static const int d1 = 41, d2 = 13, d3 = 11, d4 = 9, d5 = 76, d6 = 50;
static const int data_size = d1 * d2 * d3 * d4 * d5 * d6;
static int repeat = 1;

static const int shape_output[] = {d2, d3, d1};
static const int shape_input[] = {d4, d5, d6};
static const float shape_output_r[] = {1.f / d2, 1.f / d3, 1.f / d1};
static const float shape_input_r[] = {1.f / d4, 1.f / d5, 1.f / d6};
static const int stride_output_local[] = {d1, d1 * d2, 1};
static const int stride_output_global[] = {1, d2, d2 * d3 * d4 * d6};
static const int stride_input[] = {d2 * d3, d2 * d3 * d4 * d6 * d1, d2 * d3 * d4};

void verify(double *input, double *output) {
  int input_offset  = 2 + d1 * (2 + d2 * (2 + d3 * (2 + d4 * (0 + 2 * d5))));
  int output_offset = 2 + d2 * (2 + d3 * (2 + d4 * (2 + d6 * (2 + 0 * d1))));
  bool error = false;
  for (size_t i = 0; i < d5; i++) {
    if (input[input_offset + i * d1 * d2 * d3 * d4] != 
        output[output_offset + i * d2 * d3 * d4 * d6 * d1]) {
      printf("FAIL\n");
      error = true;
      break;
    }
  }
  if (!error) printf("PASS\n");
}

__global__ void tensor_transpose(
    const int dim_input, 
    const int dim_output, 
    const int nblocks, 
    const int tile_size,
    const int *shape_input, 
    const int *shape_output, 
    const float *shape_input_r, 
    const float *shape_output_r, 
    const int *stride_input,
    const int *stride_output_local, 
    const int *stride_output_global,
    const double *input, 
    double *output) 
{
  __shared__ double tile[TILE_SIZE];

  for (int block_idx = blockIdx.x; block_idx < nblocks; block_idx += gridDim.x) {
    int it = block_idx, im = 0, offset1 = 0;
    for (int i = 0; i < dim_input; i++) {
      im = it * shape_input_r[i];  // replace division with multiplication
      offset1 += stride_input[i] * (it - im * shape_input[i]);
      it = im;
    }

    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
      tile[i] = input[i + block_idx * tile_size];
    }

    __syncthreads();

    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
      it = i;
      int offset2 = 0, local_offset = 0;
      for (int j = 0; j < dim_output; j++) {
        im = it * shape_output_r[j];  // replace division with multiplication
        int tmp = it - im * shape_output[j];
        offset2 += stride_output_global[j] * tmp;
        local_offset += stride_output_local[j] * tmp;
        it = im;
      }
      output[offset1 + offset2] = tile[local_offset];
    }

    __syncthreads();
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  repeat = atoi(argv[1]);

  double *input = new double[data_size]();
  double *output = new double[data_size]();

  for (size_t i = 0; i < data_size; i++) {
    input[i] = i;
  }

  const int nblocks = d4 * d5 * d6;
  const int tile_size = d1 * d2 * d3;
  const int dim_output = 3;
  const int dim_input = 3;
  double *d_output, *d_input;
  int *d_shape_input, *d_shape_output;
  float *d_shape_input_r, *d_shape_output_r;
  int *d_stride_output_local, *d_stride_output_global;
  int *d_stride_input;

  cudaMalloc(&d_output, data_size * sizeof(double));
  cudaMalloc(&d_input, data_size * sizeof(double));
  cudaMalloc(&d_shape_input, dim_input * sizeof(int));
  cudaMalloc(&d_shape_input_r, dim_input * sizeof(float));
  cudaMalloc(&d_shape_output, dim_output * sizeof(int));
  cudaMalloc(&d_shape_output_r, dim_output * sizeof(float));
  cudaMalloc(&d_stride_input, dim_input * sizeof(int));
  cudaMalloc(&d_stride_output_local, dim_output * sizeof(int));
  cudaMalloc(&d_stride_output_global, dim_output * sizeof(int));

  cudaMemcpy(d_input, input, 
      data_size * sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy(d_shape_input, shape_input, 
      dim_input * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy(d_shape_input_r, shape_input_r, 
      dim_input * sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy(d_shape_output, shape_output, 
      dim_output * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy(d_shape_output_r, shape_output_r, 
      dim_output * sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy(d_stride_input, stride_input, 
      dim_input * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy(d_stride_output_local, stride_output_local, 
      dim_output * sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy(d_stride_output_global, stride_output_global, 
      dim_output * sizeof(int), cudaMemcpyHostToDevice );

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; ++i) {
    tensor_transpose<<<nblocks, NTHREADS>>>(dim_input, 
        dim_output, 
        nblocks, 
        tile_size,
        d_shape_input, 
        d_shape_output,
        d_shape_input_r,
        d_shape_output_r,
        d_stride_input,
        d_stride_output_local,
        d_stride_output_global,
        d_input, d_output);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  cudaMemcpy(output, d_output, data_size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_output);
  cudaFree(d_input);
  cudaFree(d_shape_input);
  cudaFree(d_shape_input_r);
  cudaFree(d_shape_output);
  cudaFree(d_shape_output_r);
  cudaFree(d_stride_input);
  cudaFree(d_stride_output_local);
  cudaFree(d_stride_output_global);

  verify(input, output);

  delete [] input;
  delete [] output;
  if (error) return 1;
  return 0;
}
