#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define NUM_OF_BLOCKS 1048576
#define NUM_OF_THREADS 256

/*==================================================
 References
 https://x.momo86.net/en?p=113
 https://github.com/cpc/hipcl
 ==================================================*/

inline unsigned int __byte_perm(unsigned int a, unsigned int b,
                                unsigned int s) {
  unsigned int res;
  res =
      ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff) << 8) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff) << 16) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff) << 24);
  return res;
}

sycl::half2 half_max(const sycl::half2 a, const sycl::half2 b) {
  const sycl::half2 sub = a - b;
  const unsigned sign = (*reinterpret_cast<const unsigned*>(&sub)) & 0x80008000u;
  const unsigned sw = 0x00003210 | (((sign >> 21) | (sign >> 13)) * 0x11);
  const unsigned int res = __byte_perm(*reinterpret_cast<const unsigned*>(&a), 
                                       *reinterpret_cast<const unsigned*>(&b), sw);
  return *reinterpret_cast<const sycl::half2*>(&res);
}

sycl::half half_max(const sycl::half a, const sycl::half b) {
  const sycl::half sub = a - b;
  const unsigned sign = (*reinterpret_cast<const short*>(&sub)) & 0x8000u;
  const unsigned sw = 0x00000010 | ((sign >> 13) * 0x11);
  const unsigned short res = __byte_perm(*reinterpret_cast<const short*>(&a), 
                                         *reinterpret_cast<const short*>(&b), sw);
  return *reinterpret_cast<const sycl::half*>(&res);
}

template <typename T>
void hmax(sycl::nd_item<1> &item,
          T const *__restrict const a,
          T const *__restrict const b,
          T *__restrict const r,
          const size_t size)
{
  for (size_t i = item.get_global_id(0);
              i < size; i += item.get_local_range(0) * item.get_group_range(0))
    r[i] = half_max(a[i], b[i]);
}

void generateInput(sycl::half2 * a, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    sycl::half2 temp;
    temp.x() = static_cast<float>(rand() % 922021);
    temp.y() = static_cast<float>(rand() % 922021);
    a[i] = temp;
  }
}

// compute the maximum of two values
int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  size_t size = (size_t)NUM_OF_BLOCKS * NUM_OF_THREADS;

  const size_t size_bytes = size * sizeof(sycl::half2);

  sycl::half2 * a, *b, *r;

  a = (sycl::half2*) malloc (size_bytes);
  b = (sycl::half2*) malloc (size_bytes);
  r = (sycl::half2*) malloc (size_bytes);

  // initialize input values
  srand(123); 
  generateInput(a, size);
  generateInput(b, size);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  
  sycl::half2 *d_a = sycl::malloc_device<sycl::half2>(size, q);
  q.memcpy(d_a, a, size_bytes);

  sycl::half2 *d_b = sycl::malloc_device<sycl::half2>(size, q);
  q.memcpy(d_b, b, size_bytes);

  sycl::half2 *d_r = sycl::malloc_device<sycl::half2>(size, q);

  sycl::range<1> gws (NUM_OF_BLOCKS * NUM_OF_THREADS);
  sycl::range<1> lws (NUM_OF_THREADS);

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class f16_max_v2_warmup>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        hmax<sycl::half2>(item, d_a, d_b, d_r, size);
      });
    });
  }
  q.wait();

  auto start = std::chrono::steady_clock::now();

  // run hmax2
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class f16_max_v2>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        hmax<sycl::half2>(item, d_a, d_b, d_r, size);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

  // verify
  q.memcpy(r, d_r, size_bytes).wait(); 

  bool ok = true;
  for (size_t i = 0; i < size; ++i)
  {
    sycl::float2 fa = a[i].convert<float, sycl::rounding_mode::automatic>();
    sycl::float2 fb = b[i].convert<float, sycl::rounding_mode::automatic>(); 
    sycl::float2 fr = r[i].convert<float, sycl::rounding_mode::automatic>();
    float x = fmaxf(fa.x(), fb.x());
    float y = fmaxf(fa.y(), fb.y());
    if (fabsf(fr.x() - x) > 1e-2 || fabsf(fr.y() - y) > 1e-2) {
      ok = false;
      break;
    }
  }
  printf("fp16_hmax2 %s\n", ok ?  "PASS" : "FAIL");
  if (!ok) exit(1);

  // run hmax (the size is doubled)
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class f16_max_warmup>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        hmax<sycl::half>(item, (sycl::half*)d_a, (sycl::half*)d_b, (sycl::half*)d_r, size*2);
      });
    });
  }

  q.wait();

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class f16_max>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        hmax<sycl::half>(item, (sycl::half*)d_a, (sycl::half*)d_b, (sycl::half*)d_r, size*2);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

  // verify
  q.memcpy(r, d_r, size_bytes).wait();

  ok = true;
  for (size_t i = 0; i < size; ++i)
  {
    sycl::float2 fa = a[i].convert<float, sycl::rounding_mode::automatic>();
    sycl::float2 fb = b[i].convert<float, sycl::rounding_mode::automatic>(); 
    sycl::float2 fr = r[i].convert<float, sycl::rounding_mode::automatic>();
    float x = fmaxf(fa.x(), fb.x());
    float y = fmaxf(fa.y(), fb.y());
    if (fabsf(fr.x() - x) > 1e-3 || fabsf(fr.y() - y) > 1e-3) {
      ok = false;
      break;
    }
  }

  printf("fp16_hmax %s\n", ok ?  "PASS" : "FAIL");
  if (!ok) exit(1);

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_r, q);
  free(a);
  free(b);
  free(r);

  return EXIT_SUCCESS;
}
