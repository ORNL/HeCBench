#include <chrono>
#include <iostream>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include "kernels.h"

// thread block size
#define BLOCK_SIZE 256

template <class T>
void test (sycl::queue &q, const int repeat, const int numFloats)
{
  // Initialize host data, with the first half the same as the second
  T *hostMem = (T*) malloc (sizeof(T) * numFloats);

  srand48(123);
  for (int j = 0; j < numFloats/2 ; ++j)
    hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);

  T *deviceMem = sycl::malloc_device<T>(numFloats, q);

  sycl::range<1> gws (numFloats);
  sycl::range<1> lws (BLOCK_SIZE);

  auto add1Fn = [=](sycl::nd_item<1> item) {
    Add1<T>(item, deviceMem, repeat, 10.0);
  };
  auto add2Fn = [=](sycl::nd_item<1> item) {
    Add2<T>(item, deviceMem, repeat, 10.0);
  };
  auto add4Fn = [=](sycl::nd_item<1> item) {
    Add4<T>(item, deviceMem, repeat, 10.0);
  };
  auto add8Fn = [=](sycl::nd_item<1> item) {
    Add8<T>(item, deviceMem, repeat, 10.0);
  };

  // warmup
  for (int i = 0; i < 4; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), add1Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), add2Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), add4Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), add8Fn);
    });
    q.wait();
  }

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  auto k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), add1Fn);
  });
  q.wait();
  auto k_end = std::chrono::high_resolution_clock::now();
  auto k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add1): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), add2Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add2): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), add4Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add4): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), add8Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add8): %f (s)\n", (k_time * 1e-9f));

  auto mul1Fn = [=](sycl::nd_item<1> item) {
    Mul1<T>(item, deviceMem, repeat, 1.01);
  };
  auto mul2Fn = [=](sycl::nd_item<1> item) {
    Mul2<T>(item, deviceMem, repeat, 1.01);
  };
  auto mul4Fn = [=](sycl::nd_item<1> item) {
    Mul4<T>(item, deviceMem, repeat, 1.01);
  };
  auto mul8Fn = [=](sycl::nd_item<1> item) {
    Mul8<T>(item, deviceMem, repeat, 1.01);
  };

  // warmup
  for (int i = 0; i < 4; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), mul1Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), mul2Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), mul4Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), mul8Fn);
    });
    q.wait();
  }

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), mul1Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul1): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), mul2Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul2): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), mul4Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul4): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), mul8Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul8): %f (s)\n", (k_time * 1e-9f));

  auto madd1Fn = [=](sycl::nd_item<1> item) {
    MAdd1<T>(item, deviceMem, repeat, 10.0, 0.9899);
  };
  auto madd2Fn = [=](sycl::nd_item<1> item) {
    MAdd2<T>(item, deviceMem, repeat, 10.0, 0.9899);
  };
  auto madd4Fn = [=](sycl::nd_item<1> item) {
    MAdd4<T>(item, deviceMem, repeat, 10.0, 0.9899);
  };
  auto madd8Fn = [=](sycl::nd_item<1> item) {
    MAdd8<T>(item, deviceMem, repeat, 10.0, 0.9899);
  };

  // warmup
  for (int i = 0; i < 4; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), madd1Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), madd2Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), madd4Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), madd8Fn);
    });
    q.wait();
  }

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), madd1Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd1): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), madd2Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd2): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), madd4Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd4): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), madd8Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd8): %f (s)\n", (k_time * 1e-9f));

  auto mulmadd1Fn = [=](sycl::nd_item<1> item) {
    MulMAdd1<T>(item, deviceMem, repeat, 3.75, 0.355);
  };
  auto mulmadd2Fn = [=](sycl::nd_item<1> item) {
    MulMAdd2<T>(item, deviceMem, repeat, 3.75, 0.355);
  };
  auto mulmadd4Fn = [=](sycl::nd_item<1> item) {
    MulMAdd4<T>(item, deviceMem, repeat, 3.75, 0.355);
  };
  auto mulmadd8Fn = [=](sycl::nd_item<1> item) {
    MulMAdd8<T>(item, deviceMem, repeat, 3.75, 0.355);
  };

  // warmup
  for (int i = 0; i < 4; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), mulmadd1Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), mulmadd2Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), mulmadd4Fn);
    });
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), mulmadd8Fn);
    });
    q.wait();
  }

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), mulmadd1Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd1): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), mulmadd2Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd2): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), mulmadd4Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd4): %f (s)\n", (k_time * 1e-9f));

  q.memcpy(deviceMem, hostMem, sizeof(T) * numFloats).wait();
  k_start = std::chrono::high_resolution_clock::now();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), mulmadd8Fn);
  });
  q.wait();
  k_end = std::chrono::high_resolution_clock::now();
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd8): %f (s)\n", (k_time * 1e-9f));

  free(hostMem);
  sycl::free(deviceMem, q);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  // the number of loop iterations inside kernels
  const int repeat = atoi(argv[1]);

  // a multiple of BLOCK_SIZE
  const int numFloats = 2*1024*1024;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("=== Single-precision floating-point kernels ===\n");
  test<float>(q, repeat, numFloats);

  // comment out when double-precision is not supported by a device
  printf("=== Double-precision floating-point kernels ===\n");
  test<double>(q, repeat, numFloats);

  return 0;
}
