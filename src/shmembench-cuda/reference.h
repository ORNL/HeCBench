/**
 * Host reference for the shared memory microbenchmark.
 *
 * Each thread of benchmark_shmem() owns six float4 slots of the shared memory
 * buffer (its own lane plus a stride of blockDim), initializes them, swaps them
 * TOTAL_ITERATIONS times and stores their sum. A thread never touches a slot
 * owned by another thread, so replaying the algorithm for a single block
 * reproduces the output of every block. All values are small integers, so the
 * host and the device results must match exactly.
 */

#pragma once

#include <stddef.h>
#include <vector>

struct ref_float4 { float x, y, z, w; };

static inline ref_float4 ref_init_val(int i) {
  return { (float)i, (float)(i + 11), (float)(i + 19), (float)(i + 23) };
}

static inline void ref_swap(ref_float4 &v1, ref_float4 &v2) {
  ref_float4 tmp = v2;
  v2 = v1;
  v1 = tmp;
}

// The float4 values one block of benchmark_shmem() stores, indexed by thread.
static std::vector<ref_float4> reference_block(int block_size, int iterations) {
  std::vector<ref_float4> shm(block_size * 6);

  for (int tid = 0; tid < block_size; tid++) {
    shm[tid + 0 * block_size] = ref_init_val(tid);
    shm[tid + 1 * block_size] = ref_init_val(tid + 1);
    shm[tid + 2 * block_size] = ref_init_val(tid + 3);
    shm[tid + 3 * block_size] = ref_init_val(tid + 7);
    shm[tid + 4 * block_size] = ref_init_val(tid + 13);
    shm[tid + 5 * block_size] = ref_init_val(tid + 17);
  }

  for (int j = 0; j < iterations; j++) {
    for (int tid = 0; tid < block_size; tid++) {
      ref_swap(shm[tid + 0 * block_size], shm[tid + 1 * block_size]);
      ref_swap(shm[tid + 2 * block_size], shm[tid + 3 * block_size]);
      ref_swap(shm[tid + 4 * block_size], shm[tid + 5 * block_size]);
      ref_swap(shm[tid + 1 * block_size], shm[tid + 2 * block_size]);
      ref_swap(shm[tid + 3 * block_size], shm[tid + 4 * block_size]);
    }
  }

  std::vector<ref_float4> out(block_size);
  for (int tid = 0; tid < block_size; tid++) {
    ref_float4 sum = { 0.f, 0.f, 0.f, 0.f };
    for (int s = 0; s < 6; s++) {
      const ref_float4 &v = shm[tid + s * block_size];
      sum.x += v.x;
      sum.y += v.y;
      sum.z += v.z;
      sum.w += v.w;
    }
    out[tid] = sum;
  }
  return out;
}

// Compares the num_float4 values written by the kernel against the reference
// and returns the number of mismatching values.
static int shmembench_verify(const float *c, size_t num_float4, int block_size,
                             int iterations) {
  if (block_size <= 0) return 1;

  const std::vector<ref_float4> ref = reference_block(block_size, iterations);

  int errors = 0;
  for (size_t i = 0; i < num_float4; i++) {
    const ref_float4 &r = ref[i % (size_t)block_size];
    const float *v = c + 4 * i;
    if (v[0] != r.x || v[1] != r.y || v[2] != r.z || v[3] != r.w) errors++;
  }
  return errors;
}
