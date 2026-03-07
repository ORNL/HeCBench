#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>  // iota
#include <vector>
#include <sycl/sycl.hpp>
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

  auto const& gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  for(const auto& d : gpu_devices) {
    printf("Device name: %s\n", d.get_info<sycl::info::device::name>().c_str());
  }

  const int num_devices = gpu_devices.size();
  if (num_devices <= 1) {
    printf("Warning, only one device is detected."
           "This program is supposed to execute with multiple devices.\n");
  }

  std::vector<sycl::queue> queues;

  // Alocate memory to each device. Each queue share the same context
  for (int i = 0; i < num_devices; i++) {
    queues.push_back(sycl::queue(gpu_devices[i], sycl::property::queue::in_order()));
  }

  for (long len = min_len; len <= max_len; len = len * 4) {
    std::vector<int *> device_ptr;

    for (int i = 0; i < num_devices; i++) {
      int *ptr = sycl::malloc_device<int>(len, queues[i]);
      device_ptr.push_back(ptr);
    }

    // Allocate host data, and set the value
    std::vector<int> host_ptr(len);
    std::iota(host_ptr.begin(), host_ptr.end(), 0);

    // Copy the data to the first device, and 0ed the host data
    const size_t data_size_bytes = len * sizeof(int);
    queues[0].memcpy(device_ptr[0], host_ptr.data(), data_size_bytes).wait();
    std::memset(host_ptr.data(), 0, data_size_bytes);

    auto start = std::chrono::steady_clock::now();

    // The circular exchange
    for (int n = 0; n < repeat; n++) {
      for (int i = 0; i < num_devices; i++) {
        int *src = device_ptr[i];
        int *dst = device_ptr[ ( i+1 ) % num_devices ];
        queues[i].memcpy(dst, src, data_size_bytes).wait();
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
    queues[0].memcpy(host_ptr.data(), device_ptr[0], data_size_bytes).wait();

    bool ok = true;
    for(int i = 0; i < len; i++) {
      if (host_ptr[i] != i) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) exit(1);

    for (int i = 0; i < num_devices; i++) {
      free(device_ptr[i], queues[i]);
    }
  }
  return 0;
}
