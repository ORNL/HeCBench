#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>  // iota
#include <vector>
#include <cuda.h>
#include <cstdlib>

// Send data in a circular manner in all GPU devices (non-P2P)
// Original author: Thomas Applencourt


int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <minimum copy length> <maximum copy length> <repeat>\n", argv[0]);
    return 1;
  }
  const long min_len = atol(argv[1]);
  const long max_len = atol(argv[2]);
  const int repeat = atoi(argv[3]);

  int num_devices;
  cudaGetDeviceCount(&num_devices);

  cudaDeviceProp prop;
  for(int i = 0; i < num_devices; i++) {
    cudaGetDeviceProperties(&prop, i);
    printf("Device name: %s\n", prop.name);
  }

  if (num_devices <= 1) {
    printf("Warning, only one device is detected."
           "This program is supposed to execute with multiple devices.\n");
  }

  for (long len = min_len; len <= max_len; len = len * 4) {
    std::vector<int *> device_ptr;
    const size_t data_size_bytes = len * sizeof(int);

    for (int i = 0; i < num_devices; i++) {
      cudaSetDevice(i);
      int *ptr;
      cudaMalloc((void**)&ptr, data_size_bytes);
      device_ptr.push_back(ptr);
    }

    // Allocate host data, and set the value
    std::vector<int> host_ptr(len);
    std::iota(host_ptr.begin(), host_ptr.end(), 0);

    // Copy the data to the first device, and 0ed the host data
    cudaMemcpy(device_ptr[0], host_ptr.data(), data_size_bytes, cudaMemcpyDefault);
    std::memset(host_ptr.data(), 0, data_size_bytes);

    auto start = std::chrono::steady_clock::now();

    // The circular exchange
    for (int n = 0; n < repeat; n++) {
      for (int i = 0; i < num_devices; i++) {
        int *src = device_ptr[i];
        int *dst = device_ptr[ ( i+1 ) % num_devices ];
        cudaMemcpy(dst, src, data_size_bytes, cudaMemcpyDefault);
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto time_us = time * 1e-3f / repeat;
    printf("----------------------------------------------------------------\n");
    printf("Copy length = %ld\n", len);
    printf("Average total exchange time: %f (us)\n", time_us);
    printf("Average exchange time per device: %f (us)\n", time_us / num_devices);

    // Copy back data and check for correctness
    cudaMemcpy(host_ptr.data(), device_ptr[0], data_size_bytes, cudaMemcpyDefault);

    bool ok = true;
    for(int i = 0; i < len; i++) {
      if (host_ptr[i] != i) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    for (int i = 0; i < num_devices; i++) {
      cudaFree(device_ptr[i]);
    }
  }
  if (!ok) return 1;
  return 0;
}
