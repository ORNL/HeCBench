#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define BLOCK_SIZE 256

#define ZERO_FP16 \
  sycl::bit_cast<sycl::half, unsigned short>((unsigned short)0x0000U)
#define ONE_FP16 \
  sycl::bit_cast<sycl::half, unsigned short>((unsigned short)0x3c00U)

#define ZERO_BF16 \
  sycl::bit_cast<sycl::ext::oneapi::bfloat16, unsigned short>((unsigned short)0x0000U)
#define ONE_BF16 \
  sycl::bit_cast<sycl::ext::oneapi::bfloat16, unsigned short>((unsigned short)0x3f80U)

// Reference include/dpct/atomic.hpp
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline sycl::half2 atomicAdd(sycl::half2 *addr, sycl::half2 operand) {
  auto atm = sycl::atomic_ref<unsigned, memoryOrder, memoryScope, addressSpace>(
      *reinterpret_cast<unsigned *>(addr));

  union {
    unsigned i;
    sycl::half2 h;
  } old{0}, output{0};

  while (true) {
    old.i = atm.load();
    output.h = old.h + operand;
    if (atm.compare_exchange_strong(old.i, output.i))
      break;
  }

  return output.h;
}

struct alignas(8) bfloat162
{
  sycl::ext::oneapi::bfloat16 x, y;
};

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline bfloat162 atomicAdd(bfloat162 *addr, bfloat162 operand) {
  auto atm = sycl::atomic_ref<unsigned, memoryOrder, memoryScope, addressSpace>(
      *reinterpret_cast<unsigned *>(addr));

  union {
    unsigned i;
    bfloat162 h;
  } old{0}, output{0};

  while (true) {
    old.i = atm.load();
    output.h.x = old.h.x + operand.x;
    output.h.y = old.h.y + operand.y;
    if (atm.compare_exchange_strong(old.i, output.i))
      break;
  }

  return output.h;
}

void f16AtomicOnGlobalMem(sycl::half *result, int n,
                          const sycl::nd_item<1> &item)
{
  int tid = item.get_global_id(0);
  if (tid >= n) return;
  sycl::half2 *result_v = reinterpret_cast<sycl::half2 *>(result);
  sycl::half2 val{ZERO_FP16, ONE_FP16};
  atomicAdd(&result_v[tid % BLOCK_SIZE], val);
}

void f16AtomicOnGlobalMem(sycl::ext::oneapi::bfloat16 *result, int n,
                          const sycl::nd_item<1> &item)
{
  int tid = item.get_global_id(0);
  if (tid >= n) return;
  bfloat162 *result_v = reinterpret_cast<bfloat162 *>(result);
  bfloat162 val{ZERO_BF16, ONE_BF16};
  atomicAdd(&result_v[tid % BLOCK_SIZE], val);
}

template <typename T>
void atomicCost (int nelems, int repeat)
{
  size_t result_size = sizeof(T) * BLOCK_SIZE * 2;

  T* result = (T*) malloc (result_size);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  T *d_result = (T *)sycl::malloc_device(result_size, q);
  q.memset(d_result, 0, result_size);

  sycl::range<1> lws (BLOCK_SIZE);
  sycl::range<1> gws ((nelems / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);

  auto kFn = [&](sycl::handler &cgh) {
    auto nelems_ct1 = nelems / 2;
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      f16AtomicOnGlobalMem(d_result, nelems_ct1, item);
    });
  };

  //  warmup
  q.submit(kFn);

  q.memcpy(result, d_result, result_size).wait();
  printf("Print the first two elements: 0x%04x 0x%04x\n\n", result[0], result[1]);
  printf("Print the first two elements in FLOAT32: %f %f\n\n", (float)result[0], (float)result[1]);

  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    q.submit(kFn);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of 16-bit floating-point atomic add on global memory: %f (us)\n",
          time * 1e-3f / repeat);
  free(result);
  sycl::free(d_result, q);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: total number of elements (a multiple of 2)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  assert(nelems > 0 && (nelems % 2) == 0);

  printf("\nFP16 atomic add\n");
  atomicCost<sycl::half>(nelems, repeat);

  printf("\nBF16 atomic add\n");
  atomicCost<sycl::ext::oneapi::bfloat16>(nelems, repeat);

  return 0;
}
