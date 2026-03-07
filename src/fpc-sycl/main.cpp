#include <stdio.h>      /* defines printf for tests */
#include <stdlib.h> 
#include <chrono>
#include <sycl/sycl.hpp>

typedef unsigned long ulong;

ulong * convertBuffer2Array (char *cbuffer, unsigned size, unsigned step)
{
  unsigned i,j; 
  ulong * values = NULL;
  posix_memalign((void**)&values, 1024, sizeof(ulong)*size/step);
  for (i = 0; i < size / step; i++) {
    values[i] = 0;    // Initialize all elements to zero.
  }
  for (i = 0; i < size; i += step ){
    for (j = 0; j < step; j++){
      values[i / step] += (ulong)((unsigned char)cbuffer[i + j]) << (8*j);
    }
  }
  return values;
}

unsigned my_abs ( int x )
{
  unsigned t = x >> 31;
  return (x ^ t) - t;
}

unsigned FPCCompress(ulong *values, unsigned size )
{
  unsigned compressable = 0;
  unsigned i;
  for (i = 0; i < size; i++) {
    // 000
    if(values[i] == 0){
      compressable += 1;
      continue;
    }
    // 001 010
    if(my_abs((int)(values[i])) <= 0xFF){
      compressable += 1;
      continue;
    }
    // 011
    if(my_abs((int)(values[i])) <= 0xFFFF){
      compressable += 2;
      continue;
    }
    //100  
    if(((values[i]) & 0xFFFF) == 0 ){
      compressable += 2;
      continue;
    }
    //101
    if( my_abs((int)((values[i]) & 0xFFFF)) <= 0xFF
        && my_abs((int)((values[i] >> 16) & 0xFFFF)) <= 0xFF){
      compressable += 2;
      continue;
    }
    //110
    unsigned byte0 = (values[i]) & 0xFF;
    unsigned byte1 = (values[i] >> 8) & 0xFF;
    unsigned byte2 = (values[i] >> 16) & 0xFF;
    unsigned byte3 = (values[i] >> 24) & 0xFF;
    if(byte0 == byte1 && byte0 == byte2 && byte0 == byte3){
      compressable += 1;
      continue;
    }
    //111
    compressable += 4;
  }
  return compressable;
}

unsigned f1(ulong value, bool* mask) {
  if (value == 0) {
    *mask = 1;
  } 
  return 1;
}

unsigned f2(ulong value, bool* mask) {
  if (my_abs((int)(value)) <= 0xFF) *mask = 1;
  return 1;
}

unsigned f3(ulong value, bool* mask) {
  if (my_abs((int)(value)) <= 0xFFFF) *mask = 1;
  return 2;
}

unsigned f4(ulong value, bool* mask) {
  if (((value) & 0xFFFF) == 0 ) *mask = 1;
  return 2;
}

unsigned f5(ulong value, bool* mask) {
  if ((my_abs((int)((value) & 0xFFFF))) <= 0xFF && 
      my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) 
    *mask = 1;
  return 2;
}

unsigned f6(ulong value, bool* mask) {
  unsigned byte0 = (value) & 0xFF;
  unsigned byte1 = (value >> 8) & 0xFF;
  unsigned byte2 = (value >> 16) & 0xFF;
  unsigned byte3 = (value >> 24) & 0xFF;
  if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) 
    *mask = 1;
  return 1;
}

unsigned f7(ulong value, bool* mask) {
  *mask = 1;
  return 4;
}

template<typename sycl::memory_scope MemoryScope = sycl::memory_scope::device>
static inline void atomicAdd(unsigned& val, const unsigned delta)
{
  sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, 
     MemoryScope, sycl::access::address_space::generic_space> ref(val);
  ref.fetch_add(delta);
}

void fpc_kernel (sycl::nd_item<1> &item, unsigned &compressable,
                 const ulong* values, unsigned *cmp_size)
{
  int gid = item.get_global_id(0);
  int lid = item.get_local_id(0);
  int WGS = item.get_local_range(0);

  ulong value = values[gid];
  unsigned inc;

  // 000
  if (value == 0){
    inc = 1;
  }
  // 001 010
  else if ((my_abs((int)(value)) <= 0xFF)) {
    inc = 1;
  }
  // 011
  else if ((my_abs((int)(value)) <= 0xFFFF)) {
    inc = 2;
  }
  //100  
  else if ((((value) & 0xFFFF) == 0 )) {
    inc = 2;
  }
  //101
  else if ((my_abs((int)((value) & 0xFFFF))) <= 0xFF
      && my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF ) {
    inc = 2;
  }
  //110
  else if( (((value) & 0xFF) == ((value >> 8) & 0xFF)) &&
      (((value) & 0xFF) == ((value >> 16) & 0xFF)) &&
      (((value) & 0xFF) == ((value >> 24) & 0xFF)) ) {
    inc = 1;
  } else { 
    inc = 4;
  }

  if (lid == 0) compressable = 0;
  item.barrier(sycl::access::fence_space::local_space);

  atomicAdd<sycl::memory_scope::work_group>(compressable, inc);
  item.barrier(sycl::access::fence_space::local_space);
  if (lid == WGS-1) {
    atomicAdd(cmp_size[0], compressable);
  }
}

void fpc2_kernel (sycl::nd_item<1> &item, unsigned &compressable,
                  const ulong* values, unsigned *cmp_size)
{
  int gid = item.get_global_id(0);
  int lid = item.get_local_id(0);
  int WGS = item.get_local_range(0);
  unsigned inc;

  bool m1 = 0;
  bool m2 = 0;
  bool m3 = 0;
  bool m4 = 0;
  bool m5 = 0;
  bool m6 = 0;
  bool m7 = 0;

  ulong value = values[gid];
  unsigned inc1 = f1(value, &m1);
  unsigned inc2 = f2(value, &m2);
  unsigned inc3 = f3(value, &m3);
  unsigned inc4 = f4(value, &m4);
  unsigned inc5 = f5(value, &m5);
  unsigned inc6 = f6(value, &m6);
  unsigned inc7 = f7(value, &m7);

  if (m1)
    inc = inc1;
  else if (m2)
    inc = inc2;
  else if (m3)
    inc = inc3;
  else if (m4)
    inc = inc4;
  else if (m5)
    inc = inc5;
  else if (m6)
    inc = inc6;
  else
    inc = inc7;

  if (lid == 0) compressable = 0;
  item.barrier(sycl::access::fence_space::local_space);

  atomicAdd<sycl::memory_scope::work_group>(compressable, inc);
  item.barrier(sycl::access::fence_space::local_space);
  if (lid == WGS-1) {
    atomicAdd(cmp_size[0], compressable);
  }
}

void fpc (sycl::queue &q, const ulong* values, unsigned *cmp_size_hw,
          const int values_size, const int wgs)
{
  *cmp_size_hw = 0;

  ulong *d_values = sycl::malloc_device<ulong>(values_size, q);
  q.memcpy(d_values, values, values_size * sizeof(ulong));

  unsigned *d_cmp_size = sycl::malloc_device<unsigned>(1, q);
  q.memset(d_cmp_size, 0, sizeof(unsigned));

  sycl::range<1> gws (values_size);
  sycl::range<1> lws (wgs);

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<unsigned, 0> compressable(h);
    h.parallel_for<class test1>(
      sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      fpc_kernel(item, compressable, d_values, d_cmp_size);
    });
  });

  q.memcpy(cmp_size_hw, d_cmp_size, sizeof(unsigned)).wait();
  sycl::free(d_values, q);
  sycl::free(d_cmp_size, q);
}

void fpc2 (sycl::queue &q, const ulong* values, unsigned *cmp_size_hw,
           const int values_size, const int wgs)
{
  *cmp_size_hw = 0;

  ulong *d_values = sycl::malloc_device<ulong>(values_size, q);
  q.memcpy(d_values, values, values_size * sizeof(ulong));

  unsigned *d_cmp_size = sycl::malloc_device<unsigned>(1, q);
  q.memset(d_cmp_size, 0, sizeof(unsigned));

  sycl::range<1> gws (values_size);
  sycl::range<1> lws (wgs);

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<unsigned, 0> compressable (h);
    h.parallel_for<class test2>(
      sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      fpc2_kernel(item, compressable, d_values, d_cmp_size);
    });
  });

  q.memcpy(cmp_size_hw, d_cmp_size, sizeof(unsigned)).wait();
  sycl::free(d_values, q);
  sycl::free(d_cmp_size, q);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <work-group size> <repeat>\n", argv[0]);
    return 1;
  }
  const int wgs = atoi(argv[1]); 
  const int repeat = atoi(argv[2]);

  // create the char buffer
  const int step = 4;
  const size_t size = (size_t)wgs * wgs * wgs;
  char* cbuffer = (char*) malloc (size * step);

  srand(2);
  for (int i = 0; i < size*step; i++) {
    cbuffer[i] = 0xFF << (rand() % 256);
  }

  ulong *values = convertBuffer2Array (cbuffer, size, step);
  unsigned values_size = size / step;

  // run on the host
  unsigned cmp_size = FPCCompress(values, values_size);

  unsigned cmp_size_hw; 

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  bool ok = true;
 
  // warmup
  fpc(q, values, &cmp_size_hw, values_size, wgs);

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++) {
    fpc(q, values, &cmp_size_hw, values_size, wgs);
    if (cmp_size_hw != cmp_size) {
      printf("fpc failed %u != %u\n", cmp_size_hw, cmp_size);
      ok = false;
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("fpc: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

  // warmup
  fpc2(q, values, &cmp_size_hw, values_size, wgs);

  start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++) {
    fpc2(q, values, &cmp_size_hw, values_size, wgs);
    if (cmp_size_hw != cmp_size) {
      printf("fpc2 failed %u != %u\n", cmp_size_hw, cmp_size);
      ok = false;
      break;
    }
  }

  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("fpc2: average device offload time %f (s)\n", (time * 1e-9f) / repeat);

  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(values);
  free(cbuffer);
  return 0;
}
