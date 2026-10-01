#ifndef F16ATOMIC_VERIFICATION_H
#define F16ATOMIC_VERIFICATION_H

#include <algorithm>
#include <stdio.h>

static constexpr int kBlockSize = 256;
static constexpr int kFp16MaxExactInt = 2048;
static constexpr int kBf16MaxExactInt = 256;

template <typename T>
bool verifyAtomicPairs(const T *result, int active_threads, int max_exact_int) {
  const int base_hits_per_slot = active_threads / kBlockSize;
  const int extra_slots = active_threads % kBlockSize;

  for (int i = 0; i < kBlockSize; i++) {
    const int slot_hits = base_hits_per_slot + (i < extra_slots ? 1 : 0);
    const float expected = (float)std::min(slot_hits, max_exact_int);
    const float even = (float)result[2 * i];
    const float odd = (float)result[2 * i + 1];

    if (even != 0.0f || odd != expected) {
      printf("FAIL at slot %d: expected (0.0, %f), got (%f, %f)\n",
             i, expected, even, odd);
      return false;
    }
  }

  return true;
}

#endif // F16ATOMIC_VERIFICATION_H
