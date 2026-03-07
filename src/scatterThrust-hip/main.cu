#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scatter.h>

#define CHECK_HIP(func)                                                       \
{                                                                             \
    hipError_t status = func;                                                 \
    if (status != hipSuccess) {                                               \
        printf("HIP API failed at line %d with error: %s (%d)\n",             \
               __LINE__, hipGetErrorString(status), status);                  \
    }                                                                         \
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
  CHECK_HIP( hipMalloc(&d_src, src_size_bytes) )
  CHECK_HIP( hipMalloc(&d_out, out_size_bytes) )
  CHECK_HIP( hipMalloc(&d_idx, idx_size_bytes) )
  CHECK_HIP( hipMemcpy(d_idx, h_idx, idx_size_bytes, hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(d_src, h_src, src_size_bytes, hipMemcpyHostToDevice) )

  int64_t time = 0;
  for (int i = 0; i < repeat; i++) {
    CHECK_HIP( hipMemset(d_out, 0, out_size_bytes) );
    CHECK_HIP( hipDeviceSynchronize() )
    auto start = std::chrono::steady_clock::now();
    // wrap raw pointer with a device_ptr
    thrust::device_ptr<scalar_t> t_src(d_src);
    thrust::device_ptr<scalar_t> t_out(d_out);
    thrust::device_ptr<int64_t>  t_idx(d_idx);

    thrust::scatter(t_src, t_src+num_elems, t_idx, t_out);
    CHECK_HIP( hipDeviceSynchronize() )
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of thrust::scatter: %f (us)\n", (time * 1e-3f) / repeat);

  CHECK_HIP( hipMemcpy(h_out, d_out, out_size_bytes, hipMemcpyDeviceToHost) )

  bool ok = true;
  for (int64_t i = 0; i < num_elems; i++) {
    if (h_out[i] != scalar_t(num_elems - 1 - i)) {
      ok = false;
      break;
    }
  }
  printf("%s\n\n", ok ? "PASS" : "FAIL");

  CHECK_HIP( hipFree(d_src) )
  CHECK_HIP( hipFree(d_idx) )
  CHECK_HIP( hipFree(d_out) )
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
