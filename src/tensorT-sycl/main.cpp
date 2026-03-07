#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <sycl/sycl.hpp>

#define TILE_SIZE 5900
#define NTHREADS 256

// 1,2,3,4,5,6 -> 2,3,4,6,1,5
static const int d1 = 41, d2 = 13, d3 = 11, d4 = 9, d5 = 76, d6 = 50;
static const int data_size = d1 * d2 * d3 * d4 * d5 * d6;
static int repeat = 1;

static const int shape_output[] = {d2, d3, d1};
static const int shape_input[] = {d4, d5, d6};
static const float shape_output_r[] = {1.0 / d2, 1.0 / d3, 1.0 / d1};
static const float shape_input_r[] = {1.0 / d4, 1.0 / d5, 1.0 / d6};
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
  if (error) exit(1);
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  double *d_output = sycl::malloc_device<double>(data_size, q);

  double *d_input = sycl::malloc_device<double>(data_size, q);
  q.memcpy(d_input, input, data_size * sizeof(double));

  int *d_shape_input = sycl::malloc_device<int>(dim_input, q);
  q.memcpy(d_shape_input, shape_input, dim_input * sizeof(int));

  float *d_shape_input_r = sycl::malloc_device<float>(dim_output, q);
  q.memcpy(d_shape_input_r, shape_input_r, dim_input * sizeof(float));

  int *d_shape_output = sycl::malloc_device<int>(dim_output, q);
  q.memcpy(d_shape_output, shape_output, dim_output * sizeof(int));

  float *d_shape_output_r = sycl::malloc_device<float>(dim_output, q);
  q.memcpy(d_shape_output_r, shape_output_r, dim_output * sizeof(float));

  int *d_stride_input = sycl::malloc_device<int>(dim_input, q);
  q.memcpy(d_stride_input, stride_input, dim_input * sizeof(int));

  int *d_stride_output_local = sycl::malloc_device<int>(dim_output, q);
  q.memcpy(d_stride_output_local, stride_output_local, dim_output * sizeof(int));

  int *d_stride_output_global = sycl::malloc_device<int>(dim_output, q);
  q.memcpy(d_stride_output_global, stride_output_global, dim_output * sizeof(int));

  sycl::range<1> gws (nblocks * NTHREADS);
  sycl::range<1> lws (NTHREADS);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < repeat; ++i) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<double, 1> tile(sycl::range<1>(TILE_SIZE), cgh);
      cgh.parallel_for<class transpose>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        for (int block_idx = item.get_group(0); block_idx < nblocks;
                 block_idx += item.get_group_range(0)) {
          int it = block_idx, im = 0, offset1 = 0;
          for (int i = 0; i < dim_input; i++) {
            im = it * d_shape_input_r[i];
            offset1 += d_stride_input[i] * (it - im * d_shape_input[i]);
            it = im;
          }

          for (int i = item.get_local_id(0); i < tile_size; i += item.get_local_range(0)) {
            tile[i] = d_input[i + block_idx * tile_size];
          }

          item.barrier(sycl::access::fence_space::local_space);

          for (int i = item.get_local_id(0); i < tile_size; i += item.get_local_range(0)) {
            it = i;
            int offset2 = 0, local_offset = 0;
            for (int j = 0; j < dim_output; j++) {
              im = it * d_shape_output_r[j];
              int tmp = it - im * d_shape_output[j];
              offset2 += d_stride_output_global[j] * tmp;
              local_offset += d_stride_output_local[j] * tmp;
              it = im;
            }
            d_output[offset1 + offset2] = tile[local_offset];
          }
          item.barrier(sycl::access::fence_space::local_space);
        }
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  q.memcpy(output, d_output, data_size * sizeof(double)).wait();
  sycl::free(d_output, q);
  sycl::free(d_input, q);
  sycl::free(d_shape_input, q);
  sycl::free(d_shape_input_r, q);
  sycl::free(d_shape_output, q);
  sycl::free(d_shape_output_r, q);
  sycl::free(d_stride_input, q);
  sycl::free(d_stride_output_local, q);
  sycl::free(d_stride_output_global, q);

  verify(input, output);
  delete [] input;
  delete [] output;
  return 0;
}
