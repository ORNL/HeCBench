#include <chrono>
#include <random>
#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include "kernels.h"
#include "utils.h"

#define HIP_CHECK(ans)                                                                  \
    {                                                                                   \
        gpuAssert((ans), __FILE__, __LINE__);                                           \
    }
inline void
gpuAssert(hipError_t code, const char* file, int line, bool abort = true)
{
  if(code != hipSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

template <typename Td, typename Ts>
void convert(bool isE4M3, int nelems, int niters)
{
  Ts *h_src = (Ts*) malloc (nelems * sizeof(Ts));
  Td *h_dst = (Td*) malloc (nelems * sizeof(Td));
  Td *r_dst = (Td*) malloc (nelems * sizeof(Td));

  init(isE4M3, h_src, nelems); 

  Ts *src;
  HIP_CHECK(hipMalloc((void**)&src, nelems * sizeof(Ts)));
  HIP_CHECK(hipMemcpy(src, h_src, nelems * sizeof(Ts), hipMemcpyHostToDevice));

  Td *dst;
  HIP_CHECK(hipMalloc((void**)&dst, nelems * sizeof(Td)));

  const int block_size = 256;
  const int num_blocks = (nelems + block_size - 1) / block_size;

  dim3 grid (num_blocks);
  dim3 block (block_size);

  // Warm-up run
  for (int i = 0; i < 30; i++) {
    if (isE4M3) {
      ref_fp32_cvt_e4m3<Td, Ts> <<<grid, block>>> (dst, src, nelems);
      fp32_cvt_e4m3<Td, Ts> <<<grid, block>>> (dst, src, nelems);
    }
    else {
      ref_fp32_cvt_e5m2<Td, Ts> <<<grid, block>>> (dst, src, nelems);
      fp32_cvt_e5m2<Td, Ts> <<<grid, block>>> (dst, src, nelems);
    }
  }

  HIP_CHECK(hipDeviceSynchronize());

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    if (isE4M3)
      ref_fp32_cvt_e4m3<Td, Ts> <<<grid, block>>> (dst, src, nelems);
    else
      ref_fp32_cvt_e5m2<Td, Ts> <<<grid, block>>> (dst, src, nelems);
  }
  HIP_CHECK(hipDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>
                (end - start).count() * 1.0 / niters;
  double size = (sizeof(Td) + sizeof(Ts)) * nelems;
  printf("size(GB):%.2f, average time of reference (sec):%f, BW:%f\n", size, time * 1e-9, size / time);

  HIP_CHECK(hipMemcpy(r_dst, dst, nelems * sizeof(Td), hipMemcpyDeviceToHost));

#ifdef DEBUG
  printf("Print the first 10 hex values:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f -> %x\n", h_src[i], r_dst[i]);
  }
#endif

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    if (isE4M3)
      fp32_cvt_e4m3<Td, Ts> <<<grid, block>>> (dst, src, nelems);
    else
      fp32_cvt_e5m2<Td, Ts> <<<grid, block>>> (dst, src, nelems);
  }
  HIP_CHECK(hipDeviceSynchronize());
  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>
                (end - start).count() * 1.0 / niters;
  size = (sizeof(Td) + sizeof(Ts)) * nelems;
  printf("size(GB):%.2f, average time of intrinsic (sec):%f, BW:%f\n", size, time * 1e-9, size / time);

  HIP_CHECK(hipMemcpy(h_dst, dst, nelems * sizeof(Td), hipMemcpyDeviceToHost));

#ifdef DEBUG
  printf("Print the first 10 hex values:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f -> %x\n", h_src[i], h_dst[i]);
  }
#endif

  bool ok = true;
  for (int i = 0; i < nelems; i++) {
    if (abs(int8_t(h_dst[i] - r_dst[i])) > 1) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(src));
  HIP_CHECK(hipFree(dst));
  free(h_src);
  free(h_dst);
  free(r_dst);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int niters = atoi(argv[2]);

  printf("float -> fp8 E4M3\n");
  convert<uint8_t, float>(true, nelems, niters); 

  printf("float -> fp8 E5M2\n");
  convert<uint8_t, float>(false, nelems, niters); 

  return 0;
}
