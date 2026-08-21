#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <omp.h>

const uint64_t latencyMemAccessCnt = 1000000; /* 1M total read accesses to gauge latency */
const uint32_t _2MiB = 2 * 1024 * 1024;
const uint32_t strideLen = 16; /* cacheLine size 128 Bytes, 16 words */

/* Upper bound on the number of teams used  */
const uint32_t maxTeamCount = 256;

struct LatencyNode {
  struct LatencyNode *next;
};

void initBuffer(void* buffer, uint64_t buffer_size, bool measureDeviceToDeviceLatency) {
  uint64_t n_ptrs = buffer_size / sizeof(struct LatencyNode);

  if (measureDeviceToDeviceLatency) {
    // For device-to-device latency, create and initialize pattern on device
    const int dev = omp_get_default_device();
    const int host = omp_get_initial_device();
    for (uint64_t i = 0; i < n_ptrs; i++) {
      struct LatencyNode node;
      uint64_t nextOffset = ((i + strideLen) % n_ptrs) * sizeof(struct LatencyNode);
      // Set up pattern with device addresses
      node.next = (struct LatencyNode*)((uint8_t*)buffer + nextOffset);
      int status = omp_target_memcpy(buffer, &node, sizeof(struct LatencyNode),
                                     i * sizeof(struct LatencyNode), 0,
                                     dev, host);
      if (status != 0) {
        fprintf(stderr, "omp_target_memcpy failed with status %d\n", status);
        exit(EXIT_FAILURE);
      }
    }
  } else {
    // For host-device latency, initialize pattern with host addresses
    struct LatencyNode* hostMem = (struct LatencyNode*)buffer;
    for (uint64_t i = 0; i < n_ptrs; i++) {
      hostMem[i].next = &hostMem[(i + strideLen) % n_ptrs];
    }
  }
}


double latencyPtrChaseKernel(void* data, uint64_t memAccessCnt,
                             uint32_t smCount)
{
  double latencySum = 0.0f;
  uint32_t measuredTeamCount = 0;
  struct LatencyNode *nodes = static_cast<struct LatencyNode *>(data);

  // For smCount teams, each team has memAccessCnt pointer chases
  for (uint32_t targetBlock = 0; targetBlock < smCount; ++targetBlock) {
    int executed = 0;
    auto start = std::chrono::steady_clock::now();

    // The body of a teams region is executed by one thread per team
    #pragma omp target teams num_teams(smCount) thread_limit(1) \
                             is_device_ptr(nodes) map(tofrom: executed)
    {
      if ((uint32_t)omp_get_team_num() == targetBlock) {
        executed = 1;
        struct LatencyNode *p = nodes;
        for (uint32_t i = 0; i < memAccessCnt; ++i) {
          p = p->next;
        }

        // Avoid compiler optimization: the store is never reached, but the
        // compiler cannot prove it and therefore has to keep the pointer chase
        // above. An assert() would not do, as it is compiled out when NDEBUG is
        // defined.
        if (p == nullptr) {
          nodes[0].next = nullptr;
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    if (executed) {
      auto latency =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      latencySum += latency;
      ++measuredTeamCount;
    }
  }
  if (measuredTeamCount == 0) {
    fprintf(stderr, "No OpenMP target team executed the pointer chase\n");
    exit(EXIT_FAILURE);
  }
  return latencySum /
         (memAccessCnt * measuredTeamCount); // finalLatencyPerAccessNs
}

class MemPtrChaseOperation {
  public:
    MemPtrChaseOperation() {
      // OpenMP has no portable query for the number of compute units
      int teams = 0;
      #pragma omp target teams num_teams(maxTeamCount) thread_limit(1) \
                               map(tofrom: teams)
      {
        if (omp_get_team_num() == 0) teams = omp_get_num_teams();
      }
      if (teams < 1) teams = 1;
      smCount = (uint32_t)teams;
      if (smCount > maxTeamCount) smCount = maxTeamCount;
    }
    ~MemPtrChaseOperation() = default;
    double doPtrChase(void* peerBuffer) {
      double lat =
          latencyPtrChaseKernel(peerBuffer, latencyMemAccessCnt, smCount);
      return lat;
    }
  private:
    uint32_t smCount;
};

int main() {
  const uint64_t buffer_size = _2MiB;
  const bool measureDeviceToDeviceLatency = true;

  void *buffer = omp_target_alloc(buffer_size, omp_get_default_device());
  if (buffer == nullptr) {
    fprintf(stderr, "Failed to allocate %lu bytes on the device\n",
            (unsigned long) buffer_size);
    return 1;
  }

  // initialize the buffer
  initBuffer(buffer, buffer_size, measureDeviceToDeviceLatency);

  // compute the latency of pointer chasing on the default device
  MemPtrChaseOperation mpc;

  double lat = mpc.doPtrChase(buffer);

  printf("Latency per access on device: %lf (ns)\n", lat);
  omp_target_free(buffer, omp_get_default_device());
  return 0;
}
