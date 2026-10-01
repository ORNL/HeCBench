#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>

#define GPU_CHECK(expr)                              \
  do { hipError_t e = (expr);                        \
    if (e != hipSuccess) {                           \
      fprintf(stderr, "HIP %s:%d  %s\n",             \
          __FILE__, __LINE__, hipGetErrorString(e)); \
      exit(EXIT_FAILURE); } } while (0)


const uint64_t latencyMemAccessCnt = 1000000; /* 1M total read accesses to gauge latency */
const uint32_t _2MiB = 2 * 1024 * 1024;
const uint32_t strideLen = 16; /* cacheLine size 128 Bytes, 16 words */

struct LatencyNode {
  struct LatencyNode *next;
};

void initBuffer(void* buffer, uint64_t buffer_size, bool measureDeviceToDeviceLatency) {
  uint64_t n_ptrs = buffer_size / sizeof(struct LatencyNode);

  if (measureDeviceToDeviceLatency) {
    // For device-to-device latency, create and initialize pattern on device
    for (uint64_t i = 0; i < n_ptrs; i++) {
      struct LatencyNode node;
      uint64_t nextOffset = ((i + strideLen) % n_ptrs) * sizeof(struct LatencyNode);
      // Set up pattern with device addresses
      node.next = (struct LatencyNode*)((uint8_t*)buffer + nextOffset);
      GPU_CHECK(hipMemcpy((uint8_t*)buffer + i*sizeof(struct LatencyNode),
            &node, sizeof(struct LatencyNode), hipMemcpyHostToDevice));
    }
  } else {
    // For host-device latency, initialize pattern with host addresses
    struct LatencyNode* hostMem = (struct LatencyNode*)buffer;
    for (uint64_t i = 0; i < n_ptrs; i++) {
      hostMem[i].next = &hostMem[(i + strideLen) % n_ptrs];
    }
  }
}


__global__
void ptrChasingKernel(struct LatencyNode *data,
                      const uint64_t size,
                      const uint32_t accesses,
                      const uint32_t targetBlock)
{
  if (blockIdx.x != targetBlock) return;

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

double latencyPtrChaseKernel(void* data, uint64_t size,
                             uint64_t memAccessCnt,
                             uint32_t smCount)
{
  double latencySum = 0.0f;

  // For smCount thread blocks, each block has memAccessCnt pointer chases
  for (uint32_t targetBlock = 0; targetBlock < smCount; ++targetBlock) {
    auto start = std::chrono::steady_clock::now();
    ptrChasingKernel <<<smCount, 1>>> ((struct LatencyNode*) data, size,
                                       memAccessCnt, targetBlock);
    GPU_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    latencySum += latency;
  }
  return latencySum / (memAccessCnt * smCount); // finalLatencyPerAccessNs
}

class MemPtrChaseOperation {
  public:
    MemPtrChaseOperation() {
      hipDeviceProp_t prop;
      GPU_CHECK(hipGetDeviceProperties(&prop, 0)); // device 0
      smCount = prop.multiProcessorCount;
    }
    ~MemPtrChaseOperation() = default;
    double doPtrChase(void* peerBuffer, uint64_t size) {
      double lat = latencyPtrChaseKernel(peerBuffer, size, latencyMemAccessCnt, smCount);
      return lat;
    }
  private:
    uint32_t smCount;
};

int main() {
  const uint64_t buffer_size = _2MiB;
  const bool measureDeviceToDeviceLatency = true;

  void *buffer;
  GPU_CHECK(hipMalloc((void**)&buffer, buffer_size));

  // initialize the buffer
  initBuffer(buffer, buffer_size, measureDeviceToDeviceLatency);

  // compute the latency of pointer chasing on device:0
  MemPtrChaseOperation mpc;

  double lat = mpc.doPtrChase(buffer, buffer_size);

  printf("Latency per access on device: %lf (ns)\n", lat);
  GPU_CHECK(hipFree(buffer));
  return 0;
}
