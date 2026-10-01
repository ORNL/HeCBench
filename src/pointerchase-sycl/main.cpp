#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <sycl/sycl.hpp>

const uint64_t latencyMemAccessCnt = 1000000; /* 1M total read accesses to gauge latency */
const uint32_t _2MiB = 2 * 1024 * 1024;
const uint32_t strideLen = 16; /* cacheLine size 128 Bytes, 16 words */

struct LatencyNode {
  struct LatencyNode *next;
};

void initBuffer(sycl::queue &q, void *buffer, uint64_t buffer_size,
                bool measureDeviceToDeviceLatency) {
  uint64_t n_ptrs = buffer_size / sizeof(struct LatencyNode);

  if (measureDeviceToDeviceLatency) {
    // For device-to-device latency, create and initialize pattern on device
    for (uint64_t i = 0; i < n_ptrs; i++) {
      struct LatencyNode node;
      uint64_t nextOffset = ((i + strideLen) % n_ptrs) * sizeof(struct LatencyNode);
      // Set up pattern with device addresses
      node.next = (struct LatencyNode*)((uint8_t*)buffer + nextOffset);
      q.memcpy((uint8_t *)buffer + i * sizeof(struct LatencyNode), &node,
                sizeof(struct LatencyNode));
    }
  } else {
    // For host-device latency, initialize pattern with host addresses
    struct LatencyNode* hostMem = (struct LatencyNode*)buffer;
    for (uint64_t i = 0; i < n_ptrs; i++) {
      hostMem[i].next = &hostMem[(i + strideLen) % n_ptrs];
    }
  }
}

void ptrChasingKernel(struct LatencyNode *data,
                      const uint64_t size,
                      const uint32_t accesses,
                      const uint32_t targetBlock,
                      sycl::nd_item<1> &item)
{
  if (item.get_group(0) != targetBlock) return;

  struct LatencyNode *p = data;
  for (auto i = 0; i < accesses; ++i) {
    p = p->next;
  }

  // Avoid compiler optimization: the store is never reached, but the compiler
  // cannot prove it and therefore has to keep the pointer chase above. An
  // assert() would not do, as it is compiled out when NDEBUG is defined.
  if (p == nullptr) {
    data[0].next = nullptr;
  }
}

double latencyPtrChaseKernel(sycl::queue &q, void *data, uint64_t size,
                             uint64_t memAccessCnt,
                             uint32_t smCount)
{
  double latencySum = 0.0f;

  // For smCount thread blocks, each block has memAccessCnt pointer chases
  for (uint32_t targetBlock = 0; targetBlock < smCount; ++targetBlock) {
    auto start = std::chrono::steady_clock::now();
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(smCount), sycl::range<1>(1)),
                   [=](sycl::nd_item<1> item) {
      ptrChasingKernel((struct LatencyNode *)data, size,
                       memAccessCnt, targetBlock, item);
    }).wait();
    auto end = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    latencySum += latency;
  }
  return latencySum / (memAccessCnt * smCount); // finalLatencyPerAccessNs
}

class MemPtrChaseOperation {
  public:
    MemPtrChaseOperation(sycl::queue &q) {
      smCount = q.get_device().get_info<sycl::info::device::max_compute_units>();
    }
    ~MemPtrChaseOperation() = default;
    double doPtrChase(sycl::queue &q, void* peerBuffer, uint64_t size) {
      double lat = latencyPtrChaseKernel(q, peerBuffer, size, latencyMemAccessCnt, smCount);
      return lat;
    }
  private:
    uint32_t smCount;
};

int main() {
  const uint64_t buffer_size = _2MiB;
  const bool measureDeviceToDeviceLatency = true;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  void *buffer = (void *)sycl::malloc_device(buffer_size, q);

  // initialize the buffer
  initBuffer(q, buffer, buffer_size, measureDeviceToDeviceLatency);

  // compute the latency of pointer chasing on device:0
  MemPtrChaseOperation mpc (q);

  double lat = mpc.doPtrChase(q, buffer, buffer_size);

  printf("Latency per access on device: %lf (ns)\n", lat);
  sycl::free(buffer, q);
  return 0;
}
