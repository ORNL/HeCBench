#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/scatter.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = func;                                                 \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}


template<typename scalar_t>
void scatter(int64_t num_elems, int repeat) {
  int64_t src_size_bytes = num_elems * sizeof(scalar_t);
  int64_t idx_size_bytes = num_elems * sizeof(int64_t);
  int64_t out_size_bytes = num_elems * sizeof(scalar_t);

  int64_t *h_idx = (int64_t*) malloc (idx_size_bytes);
  scalar_t *h_src = (scalar_t*) malloc (src_size_bytes);
  scalar_t *h_out = (scalar_t*) malloc (out_size_bytes);
  srand(123);
  for (int64_t i = 0; i < num_elems; i++) {
    h_idx[i] = num_elems - 1 - i;
    h_src[i] = i;
  }

  scalar_t *d_src, *d_out;
  int64_t *d_idx;
  CHECK_CUDA( cudaMalloc(&d_src, src_size_bytes) )
  CHECK_CUDA( cudaMalloc(&d_out, out_size_bytes) )
  CHECK_CUDA( cudaMalloc(&d_idx, idx_size_bytes) )
  CHECK_CUDA( cudaMemcpy(d_idx, h_idx, idx_size_bytes, cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_src, h_src, src_size_bytes, cudaMemcpyHostToDevice) )

  int64_t time = 0;
  for (int i = 0; i < repeat; i++) {
    CHECK_CUDA( cudaMemset(d_out, 0, out_size_bytes) );
    CHECK_CUDA( cudaDeviceSynchronize() )
    auto start = std::chrono::steady_clock::now();
    // wrap raw pointer with a device_ptr
    thrust::device_ptr<scalar_t> t_src(d_src);
    thrust::device_ptr<scalar_t> t_out(d_out);
    thrust::device_ptr<int64_t>  t_idx(d_idx);

    thrust::scatter(t_src, t_src+num_elems, t_idx, t_out);
    CHECK_CUDA( cudaDeviceSynchronize() )
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of thrust::scatter: %f (us)\n", (time * 1e-3f) / repeat);

  CHECK_CUDA( cudaMemcpy(h_out, d_out, out_size_bytes, cudaMemcpyDeviceToHost) )

  bool ok = true;
  for (int64_t i = 0; i < num_elems; i++) {
    if (h_out[i] != scalar_t(num_elems - 1 - i)) {
      ok = false;
      break;
    }
  }
  printf("%s\n\n", ok ? "PASS" : "FAIL");

  CHECK_CUDA( cudaFree(d_src) )
  CHECK_CUDA( cudaFree(d_idx) )
  CHECK_CUDA( cudaFree(d_out) )
  free(h_src);
  free(h_idx);
  free(h_out);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int64_t num_elements = atol(argv[1]);
  const int repeat = atoi(argv[2]);
  printf("INT8 scatter\n");
  scatter<int8_t>(num_elements, repeat);
  printf("INT16 scatter\n");
  scatter<int16_t>(num_elements, repeat);
  printf("INT32 scatter\n");
  scatter<int32_t>(num_elements, repeat);
  printf("INT64 scatter\n");
  scatter<int64_t>(num_elements, repeat);
  printf("FP32 scatter\n");
  scatter<float>(num_elements, repeat);
  printf("FP64 scatter\n");
  scatter<double>(num_elements, repeat);
  if (!ok) return 1;
  return 0;
}
