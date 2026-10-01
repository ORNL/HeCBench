#pragma once

#include <cassert>
#include <hip/hip_runtime.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>

#define NUM_BLOCKS(n, block_size) (((n) + (block_size) - 1) / (block_size))
#define THREAD_PER_BLOCK 128

#include <chrono>

// Number of nanoseconds per second; timing consumers divide hrClock()
// differences by this to obtain seconds.
constexpr double NS_PER_SEC = 1e9;

// Returns the number of nanoseconds elapsed since the first call.
inline double hrClock() {
    static const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now() - t0).count();
}

// does not use uint64_t in cstdint since it's not supported by atomicCAS
using uint64 = unsigned long long int;
using uint32 = unsigned int;

// Ordered CAS/load helpers for the lock-free chained hash table in rewrite.cu.
// Several call sites publish a node into a bucket chain by CAS-ing its "next"
// pointer *after* having written the node's payload (.val) with a plain store.
// The default atomicCAS is relaxed and acts as no memory fence, so on the
// weakly-ordered AMDGPU a concurrent reader that follows the published "next"
// link may still observe a stale payload. Pairing an acq_rel CAS on the writer
// with an acquire load on the reader closes that reordering window.
__device__ __forceinline__ int atomicCASAcqRel(int *addr, int expected, int desired) {
    int expected_val = expected;
    __hip_atomic_compare_exchange_strong(addr, &expected_val, desired,
                                         __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
                                         __HIP_MEMORY_SCOPE_AGENT);
    // Mirrors atomicCAS: returns the value previously stored at addr.
    return expected_val;
}

__device__ __forceinline__ int atomicLoadAcquire(const int *addr) {
    return __hip_atomic_load(addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

// ROCm 6.x does not ship a __syncwarp intrinsic (only the *_sync warp
// builtins gated by HIP_ENABLE_WARP_SYNC_BUILTINS). truth.cuh needs it to make
// a lane-0 shared-memory store visible to the rest of the warp. On AMDGPU
// wavefront lanes execute in lockstep, so at this uniform program point a
// workgroup-scope fence (LDS visibility) plus a wave reconvergence/scheduling
// barrier is a correct stand-in.
#ifndef __HIP_SYNCWARP_SHIM
#define __HIP_SYNCWARP_SHIM
__device__ __forceinline__ void __syncwarp(unsigned long long mask = ~0ull) {
    (void)mask;
    __threadfence_block();
    __builtin_amdgcn_wave_barrier();
}
#endif

// for static_assert false
template <class... T>
constexpr bool always_false = false;

inline int AigNodeID(int lit) {return lit >> 1;}
inline int AigNodeIsComplement(int lit) {return lit & 1;}
inline unsigned invertConstTrueFalse(unsigned lit) {
    // swap 0 and 1
    return lit < 2 ? 1 - lit : lit;
}


namespace dUtils {

__host__ __device__ __forceinline__ int AigNodeID(int lit) {return lit >> 1;}
__host__ __device__ __forceinline__ int AigNodeIsComplement(int lit) {return lit & 1;}
__host__ __device__ __forceinline__ int AigIsNode(int nodeId, int nPIs) {return nodeId > nPIs;} // considering const 1
__host__ __device__ __forceinline__ int AigIsPIConst(int nodeId, int nPIs) {return nodeId <= nPIs;}
__host__ __device__ __forceinline__ int AigNodeLitCond(int nodeId, int complement) {
    return (int)(((unsigned)nodeId << 1) | (unsigned)(complement != 0));
}
__host__ __device__ __forceinline__ int AigNodeNotCond(int lit, int complement) {
    return (int)((unsigned)lit ^ (unsigned)(complement != 0));
}
__host__ __device__ __forceinline__ int AigNodeNot(int lit) {return lit ^ 1;}
__host__ __device__ __forceinline__ int AigNodeIDDebug(int lit, int nPIs, int nPOs) {
    int id = dUtils::AigNodeID(lit);
    return dUtils::AigIsNode(id, nPIs) ? (id + nPOs) : id;
}
__host__ __device__ __forceinline__ int TruthWordNum(int nVars) {return nVars <= 5 ? 1 : (1 << (nVars - 5));}
__host__ __device__ __forceinline__ int Truth6WordNum(int nVars) {return nVars <= 6 ? 1 : (1 << (nVars - 6));}

// unary thrust functor
template <typename ValueT, typename MaskT>
struct getValueFilteredByMask {
    const MaskT maskTrue;

    getValueFilteredByMask() : maskTrue(1) {}
    getValueFilteredByMask(MaskT _maskTrue) : maskTrue(_maskTrue) {}
    
    __host__ __device__
    ValueT operator()(const thrust::tuple<ValueT, MaskT> &elem) const { 
        return thrust::get<1>(elem) == maskTrue ? thrust::get<0>(elem) : 0;
    }
};

template <typename IdT, typename LevelT>
struct decreaseLevelIds {
    __host__ __device__
    bool operator()(const thrust::tuple<IdT, LevelT> &e1, const thrust::tuple<IdT, LevelT> &e2) const {
        return thrust::get<1>(e1) == thrust::get<1>(e2) ? 
            (thrust::get<0>(e1) > thrust::get<0>(e2)) : (thrust::get<1>(e1) > thrust::get<1>(e2));
        // return thrust::get<1>(e1) > thrust::get<1>(e2);
    }
};

template <typename IdT, typename LevelT>
struct decreaseLevels {
    __host__ __device__
    bool operator()(const thrust::tuple<IdT, LevelT> &e1, const thrust::tuple<IdT, LevelT> &e2) const {
        return thrust::get<1>(e1) > thrust::get<1>(e2);
    }
};

template <typename IdT, typename LevelT>
struct decreaseLevelsPerm {
    __host__ __device__
    bool operator()(const thrust::tuple<IdT, LevelT, int> &e1, const thrust::tuple<IdT, LevelT, int> &e2) const {
        return thrust::get<1>(e1) == thrust::get<1>(e2) ? 
            (thrust::get<2>(e1) > thrust::get<2>(e2)) : (thrust::get<1>(e1) > thrust::get<1>(e2));
    }
};

template <typename T>
struct isMinusOne { 
    __host__ __device__
    bool operator()(const T &elem) {
        return elem == -1;
    }
};

template <typename T>
struct isOne { 
    __host__ __device__
    bool operator()(const T &elem) {
        return elem == 1;
    }
};

template <typename T>
struct isNotOne { 
    __host__ __device__
    bool operator()(const T &elem) {
        return elem != 1;
    }
};

template <typename T, T val>
struct equalsVal {
    __host__ __device__
    bool operator()(const T &elem) {
        return elem == val;
    }
};

template <typename T, T val>
struct notEqualsVal {
    __host__ __device__
    bool operator()(const T &elem) {
        return elem != val;
    }
};

template <typename T, T val>
struct greaterThanVal {
    __host__ __device__
    bool operator()(const T &elem) {
        return elem > val;
    }
};

template <typename T, T val>
struct greaterThanEqualsValInt {
    __host__ __device__
    int operator()(const T &elem) {
        return (elem >= val ? 1 : 0);
    }
};

template <typename T, T val>
struct lessThanVal {
    __host__ __device__
    bool operator()(const T &elem) {
        return elem < val;
    }
};

struct sameNodeID {
    __host__ __device__
    bool operator()(const int &lhs, const int &rhs) const {
        return (lhs >> 1) == (rhs >> 1);
    }
};

template <typename T>
struct isNodeLit {
    const int nPIs;

    isNodeLit() = delete;
    isNodeLit(const int _nPIs) : nPIs(_nPIs) {}

    __host__ __device__
    bool operator()(const T &elem) {
        return (elem >> 1) > nPIs;
    }
};

template <typename T>
struct isPIConstLit {
    const int nPIs;

    isPIConstLit() = delete;
    isPIConstLit(const int _nPIs) : nPIs(_nPIs) {}

    __host__ __device__
    bool operator()(const T &elem) {
        return (elem >> 1) <= nPIs;
    }
};

struct getNodeID {
    __host__ __device__
    int operator()(const int elem) {
        return elem >> 1;
    }
};

const int AigConst1 = 0;
const int AigConst0 = 1;
} // namespace dUtils

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      fprintf(stderr,"GPUassert: %s,\nat %s, line %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuChkStackOverflow(ans) { gpuAssertStack((ans), __FILE__, __LINE__); }
inline void gpuAssertStack(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess)
   {
      fprintf(stderr,"GPUassert: %s,\nat %s, line %d\n", hipGetErrorString(code), file, line);
      fprintf(stderr, "This is most likely due to insufficient HIP call stack size. Try to increase hipLimitStackSize.\n");
      if (abort) exit(code);
   }
}
