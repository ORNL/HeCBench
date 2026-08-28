//
// Copyright 2004-present Facebook. All Rights Reserved.
//

#include <stdlib.h>
#include <assert.h>
#include <iterator>
#include <vector>
#include <chrono>
#include <cuda.h>

using namespace std;

#define WARP_SIZE 32
#define MASK 0xffffffff

/// A simple pair type for CUDA device usage
template <typename K, typename V>
struct Pair {
  __host__ __device__ __forceinline__ Pair() {}

  __host__ __device__ __forceinline__ Pair(K key, V value)
      : k(key), v(value) {}

  __host__ __device__ __forceinline__ bool
  operator==(const Pair<K, V>& rhs) const {
    return (k == rhs.k) && (v == rhs.v);
  }

  __host__ __device__ __forceinline__ bool
  operator!=(const Pair<K, V>& rhs) const {
    return !operator==(rhs);
  }

  __host__ __device__ __forceinline__ bool
  operator<(const Pair<K, V>& rhs) const {
    return (k < rhs.k) || ((k == rhs.k) && (v < rhs.v));
  }

  __host__ __device__ __forceinline__ bool
  operator>(const Pair<K, V>& rhs) const {
    return (k > rhs.k) || ((k == rhs.k) && (v > rhs.v));
  }

  K k;
  V v;
};

/**
   Extract a single bit at `pos` from `val`
*/

__device__ __forceinline__ int getBit(int val, int pos) {
  return (val >> pos) & 0x1;
}

/**
   Return the current thread's lane in the warp
*/
__device__ __forceinline__ int getLaneId() {
  return threadIdx.x % WARP_SIZE;
}

template <typename T>
struct GreaterThan {
  static __device__ __forceinline__ bool compare(const T lhs, const T rhs) {
    return (lhs > rhs);
  }
};

template <typename T>
struct LessThan {
  static __device__ __forceinline__ bool compare(const T lhs, const T rhs) {
    return (lhs < rhs);
  }
};

template <typename T>
__device__ __forceinline__ T
shfl_xor(const T val, int laneMask, int width = WARP_SIZE) {
  return __shfl_xor_sync(MASK, val, laneMask, width);
}

template <typename K, typename V>
__device__ __forceinline__ Pair<K, V>
shfl_xor(const Pair<K, V>& p, int laneMask, int width = WARP_SIZE) {
  return Pair<K, V>(__shfl_xor_sync(MASK, p.k, laneMask, width),
                    __shfl_xor_sync(MASK, p.v, laneMask, width));
}

template <typename T, typename Comparator>
__device__ __forceinline__ T shflSwap(const T x, int mask, int dir) {
  T y = shfl_xor(x, mask);
  return Comparator::compare(x, y) == dir ? y : x;
}

/// Defines a bitonic sort network to exchange 'V' according to
/// `SWAP()`'s compare and exchange mechanism across the warp, ordered
/// according to the comparator `comp`. In other words, if `comp` is
/// `GreaterThan<T>`, then lane 0 will contain the highest `val`
/// presented across the warp
///
/// See also 
/// http://on-demand.gputechconf.com/gtc/2013/presentations/S3174-Kepler-Shuffle-Tips-Tricks.pdf
template <typename T, typename Comparator>
__device__ T warpBitonicSort(T val) {
  const int laneId = getLaneId();
  // 2
  val = shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 1) ^ getBit(laneId, 0));

  // 4
  val = shflSwap<T, Comparator>(val, 0x02, getBit(laneId, 2) ^ getBit(laneId, 1));
  val = shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 2) ^ getBit(laneId, 0));

  // 8
  val = shflSwap<T, Comparator>(val, 0x04, getBit(laneId, 3) ^ getBit(laneId, 2));
  val = shflSwap<T, Comparator>(val, 0x02, getBit(laneId, 3) ^ getBit(laneId, 1));
  val = shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 3) ^ getBit(laneId, 0));

  // 16
  val = shflSwap<T, Comparator>(val, 0x08, getBit(laneId, 4) ^ getBit(laneId, 3));
  val = shflSwap<T, Comparator>(val, 0x04, getBit(laneId, 4) ^ getBit(laneId, 2));
  val = shflSwap<T, Comparator>(val, 0x02, getBit(laneId, 4) ^ getBit(laneId, 1));
  val = shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 4) ^ getBit(laneId, 0));

  // 32
  val = shflSwap<T, Comparator>(val, 0x10, getBit(laneId, 4));
  val = shflSwap<T, Comparator>(val, 0x08, getBit(laneId, 3));
  val = shflSwap<T, Comparator>(val, 0x04, getBit(laneId, 2));
  val = shflSwap<T, Comparator>(val, 0x02, getBit(laneId, 1));
  val = shflSwap<T, Comparator>(val, 0x01, getBit(laneId, 0));

  return val;
}

/// Determine if two warp threads have the same value (a collision).
template <typename T>
__device__ __forceinline__ bool warpHasCollision(T val) {
  // -sort all values
  // -compare our lower neighbor's value against ourselves (excepting
  //  the first lane)
  // -if any lane as a difference of 0, there is a duplicate
  //  (excepting the first lane)
  val = warpBitonicSort<T, LessThan<T>>(val);
  const T lower = __shfl_up_sync(MASK, val, 1);

  // Shuffle for lane 0 will present its same value, so only
  // subsequent lanes will detect duplicates
  const bool dup = (lower == val) && (getLaneId() != 0);
  return (__any_sync(MASK, dup) != 0);
}

/// Determine if two warp threads have the same value (a collision),
/// and returns a bitmask of the lanes that are known to collide with
/// other lanes. Not all lanes that are mutually colliding return a
/// bit; all lanes with a `1` bit are guaranteed to collide with a
/// lane with a `0` bit, so the mask can be used to serialize
/// execution for lanes that collide with others.
/// (mask | (mask >> 1)) will yield all mutually colliding lanes.
template <typename T>
__device__ __forceinline__ unsigned int warpCollisionMask(T val) {
  // -sort all (lane, value) pairs on value
  // -compare our lower neighbor's value against ourselves (excepting
  //  the first lane)
  // -if any lane as a difference of 0, there is a duplicate
  //  (excepting the first lane)
  // -shuffle sort (originating lane, dup) pairs back to the original
  //  lane and report
  Pair<T, int> pVal(val, getLaneId());

  pVal = warpBitonicSort<Pair<T, int>, LessThan<Pair<T, int> > >(pVal);

  // If our neighbor is the same as us, we know our thread's value is
  // duplicated. All except for lane 0, since shfl will present its
  // own value (and if lane 0's value is duplicated, lane 1 will pick
  // that up)
  const unsigned long lower = __shfl_up_sync(MASK, pVal.k, 1);
  Pair<int, bool> dup(pVal.v, (lower == pVal.k) && (getLaneId() != 0));

  // Sort back based on lane ID so each thread originally knows
  // whether or not it duplicated
  dup = warpBitonicSort<Pair<int, bool>,
                        LessThan<Pair<int, bool> > >(dup);
  return __ballot_sync(MASK, dup.v);
}

__device__ int hasDuplicate[32];

__global__ void checkDuplicates(int num, const int* v) {
  hasDuplicate[threadIdx.x] = (int) warpHasCollision(v[threadIdx.x]);
}

__device__ unsigned int duplicateMask;

__global__ void checkDuplicateMask(int num, const int* v) {
  unsigned int mask = warpCollisionMask(v[threadIdx.x]);
  if (threadIdx.x == 0) {
    duplicateMask = mask;
  }
}

vector<int> checkDuplicates(const vector<int>& v) {
  int* devSet = NULL;
  cudaMalloc(&devSet, v.size() * sizeof(int));
  cudaMemcpy(devSet, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice);

  checkDuplicates<<<1, 32>>>(v.size(), devSet);

  vector<int> hasDuplicates(32, false);
  cudaMemcpyFromSymbol(hasDuplicates.data(),
                       hasDuplicate, sizeof(int) * 32, 0,
                       cudaMemcpyDeviceToHost);
  cudaFree(devSet);

  return hasDuplicates;
}

unsigned int checkDuplicateMask(const vector<int>& v) {
  int* devSet = NULL;
  cudaMalloc(&devSet, v.size() * sizeof(int));
  cudaMemcpy(devSet, v.data(), v.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  checkDuplicateMask<<<1, 32>>>(v.size(), devSet);

  unsigned int mask = 0;

  cudaMemcpyFromSymbol(&mask,
                       duplicateMask, sizeof(unsigned int), 0,
                       cudaMemcpyDeviceToHost);
  cudaFree(devSet);

  return mask;
}

void test_collision(const int ND) {
  for (int numDups = 0; numDups < ND; ++numDups) {

    vector<int> v;
    for (int i = 0; i < ND - numDups; ++i) {
      int r = 0;

      while (true) {
        r = rand();

        bool found = false;
        for (unsigned int j = 0; j < v.size(); ++j) {
          if (v[j] == r) {
            found = true;
            break;
          }
        }

        if (!found) {
          break;
        }
      }

      v.push_back(r);
    }

    for (int i = 0; i < numDups; ++i) {
      v.push_back(v[0]);
    }

    assert(ND == v.size());
    auto dupCheck = checkDuplicates(v);

    for (auto dup : dupCheck) {
      assert((numDups > 0) == dup);
    }
  }
}

void test_collisionMask(const int ND) {
  for (int numDups = 0; numDups < ND; ++numDups) {
    vector<int> v;
    for (int i = 0; i < ND - numDups; ++i) {
      int r = 0;

      while (true) {
        r = rand();

        bool found = false;
        for (unsigned int j = 0; j < v.size(); ++j) {
          if (v[j] == r) {
            found = true;
            break;
          }
        }

        if (!found) {
          break;
        }
      }

      v.push_back(r);
    }

    for (int i = 0; i < numDups; ++i) {
      v.push_back(v[0]);
    }

    assert (ND == v.size());

    auto mask = checkDuplicateMask(v);
    auto expected = numDups > 0 ? 0xffffffffU << (ND - numDups) : 0;
    if (expected != mask) {
      printf("Error: numDups=%d expected=%x mask=%x\n", numDups, expected, mask);
      break;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  srand(123);
  const int num_dup = 32;
  const int repeat = atoi(argv[1]);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) 
    test_collision(num_dup);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the function test_collision: %f (us)\n",
         time * 1e-3f / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) 
    test_collisionMask(num_dup);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the function test_collisionMask: %f (us)\n",
         time * 1e-3f / repeat);

  return 0;
}
