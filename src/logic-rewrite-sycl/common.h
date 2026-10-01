#pragma once

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>
#include <sycl/sycl.hpp>
#include <cassert>

#include <functional>

#define NUM_BLOCKS(n, block_size) (((n) + (block_size) - 1) / (block_size))
#define THREAD_PER_BLOCK 128

#include <chrono>
#include <ctime>

// Number of nanoseconds per second; timing consumers divide hrClock()
// differences by this to obtain seconds.
constexpr double NS_PER_SEC = 1e9;

// Wall-clock timer built on std::chrono::steady_clock. Returns the number of
// nanoseconds elapsed since the first call.
inline double hrClock() {
    static const std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

// does not use uint64_t in cstdint since it's not supported by atomicCAS
using uint64 = unsigned long long int;
using uint32 = unsigned int;

// for static_assert false
template <class... T>
constexpr bool always_false = false;

// ---------------------------------------------------------------------------
// These are built directly on sycl::atomic_ref / sycl 2020 free functions
// ---------------------------------------------------------------------------
namespace nsycl {

template <typename T, typename U>
inline T atomic_fetch_add(T *addr, U operand) {
    return sycl::atomic_ref<T, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::generic_space>(*addr)
        .fetch_add(static_cast<T>(operand));
}

template <typename T, typename U>
inline T atomic_fetch_sub(T *addr, U operand) {
    return sycl::atomic_ref<T, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::generic_space>(*addr)
        .fetch_sub(static_cast<T>(operand));
}

template <typename T, typename U>
inline T atomic_fetch_max(T *addr, U operand) {
    return sycl::atomic_ref<T, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::generic_space>(*addr)
        .fetch_max(static_cast<T>(operand));
}

// Mimics CUDA atomicCAS: atomically
// compares *addr with expected, storing desired on success, and returns the
// value that was previously stored at addr.
//
// Uses acq_rel ordering: several call sites publish a node into a lock-free
// linked list by CAS-ing a "next" pointer after having written the node's
// payload with a plain store. The CAS must therefore act as a release so that
// a concurrent reader which observes the published pointer (via atomic_load,
// below) also observes the payload. On weakly-ordered GPUs (e.g. AMD) a relaxed
// CAS would let the reader see the link before the payload and dereference a
// stale value. (seq_cst is not supported on AMDGPU, so acq_rel is used.)
template <typename T, typename U, typename V>
inline T atomic_compare_exchange(T *addr, U expected, V desired) {
    T expected_value = static_cast<T>(expected);
    sycl::atomic_ref<T, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space>
        ref(*addr);
    ref.compare_exchange_strong(expected_value, static_cast<T>(desired));
    return expected_value;
}

// Acquire-load, pairing with the release in atomic_compare_exchange above so
// that observing a published pointer also makes the pointed-to payload visible.
template <typename T>
inline T atomic_load(const T *addr) {
    return sycl::atomic_ref<T, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::generic_space>(
               *const_cast<T *>(addr))
        .load(sycl::memory_order::acquire);
}

// fills first[i] = init + i * step on the
// given in-order queue. Ordering with subsequent queue work is preserved.
template <typename It, typename T>
inline void iota(sycl::queue &q, It first, It last, T init, T step = T(1)) {
    const size_t n = static_cast<size_t>(last - first);
    if (n == 0)
        return;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        first[i] = init + static_cast<T>(i[0]) * step;
    });
}

// Overload for the default (init = 0, step = 1) form.
template <typename It>
inline void iota(sycl::queue &q, It first, It last) {
    using T = std::remove_cv_t<std::remove_reference_t<decltype(*first)>>;
    iota(q, first, last, T(0), T(1));
}

// Process-wide native in-order SYCL queue
inline sycl::queue &queue() {
    static sycl::queue q{sycl::default_selector_v,
                         sycl::property::queue::in_order()};
    return q;
}

// ---- stencil compactions
// The predicate is applied to the stencil (mask) element.
// Implemented with zip iterators + discard_iterator. ----

// Copy first[i] into result where pred(mask[i]) is true; returns result end.
template <typename It1, typename It2, typename It3, typename Pred>
It3 copy_if(sycl::queue &q, It1 first, It1 last, It2 mask, It3 result,
            Pred pred) {
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    const auto n = last - first;
    auto zb = oneapi::dpl::make_zip_iterator(first, mask);
    auto ze = oneapi::dpl::make_zip_iterator(last, mask + n);
    auto zr = oneapi::dpl::make_zip_iterator(result,
                                             oneapi::dpl::discard_iterator());
    auto zend = oneapi::dpl::copy_if(
        policy, zb, ze, zr,
        [pred](const auto &t) { return (bool)pred(std::get<1>(t)); });
    return std::get<0>(zend.base());
}

// Remove (in place) elements first[i] where pred(mask[i]) is true; keeps the
// rest compacted at the front; returns the new logical end.
template <typename It1, typename It2, typename Pred>
It1 remove_if(sycl::queue &q, It1 first, It1 last, It2 mask, Pred pred) {
    using V = typename std::iterator_traits<It1>::value_type;
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    const auto n = last - first;
    V *tmp = sycl::malloc_device<V>(n, q);
    auto zb = oneapi::dpl::make_zip_iterator(first, mask);
    auto ze = oneapi::dpl::make_zip_iterator(last, mask + n);
    auto zr = oneapi::dpl::make_zip_iterator(tmp,
                                             oneapi::dpl::discard_iterator());
    auto zend = oneapi::dpl::copy_if(
        policy, zb, ze, zr,
        [pred](const auto &t) { return !(bool)pred(std::get<1>(t)); });
    const auto kept = std::get<0>(zend.base()) - tmp;
    oneapi::dpl::copy(policy, tmp, tmp + kept, first);
    sycl::free(tmp, q);
    return first + kept;
}

// Replace first[i] with new_value where pred(mask[i]) is true.
template <typename It1, typename It2, typename Pred, typename T>
void replace_if(sycl::queue &q, It1 first, It1 last, It2 mask, Pred pred,
                T new_value) {
    using V = typename std::iterator_traits<It1>::value_type;
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    oneapi::dpl::transform(policy, first, last, mask, first,
                           [pred, new_value](const V &val, const auto &m) {
                               return pred(m) ? static_cast<V>(new_value) : val;
                           });
}

template <typename T>
class shared_scalar {
    T *ptr_ = nullptr;
    T init_val_;
    bool has_init_val_;
    void ensure() {
        if (!ptr_) {
            ptr_ = sycl::malloc_shared<T>(1, queue());
            *ptr_ = has_init_val_ ? init_val_ : T{};
        }
    }

  public:
    shared_scalar() : init_val_(T{}), has_init_val_(false) {}
    explicit shared_scalar(T v) : init_val_(v), has_init_val_(true) {}
    void init() { ensure(); }
    T *get_ptr() {
        ensure();
        return ptr_;
    }
    T &operator[](size_t) {
        ensure();
        return *ptr_;
    }
};

} // namespace nsycl

inline int AigNodeID(int lit) {return lit >> 1;}
inline int AigNodeIsComplement(int lit) {return lit & 1;}
inline unsigned invertConstTrueFalse(unsigned lit) {
    // swap 0 and 1
    return lit < 2 ? 1 - lit : lit;
}


namespace dUtils {

inline int AigNodeID(int lit) { return lit >> 1; }
inline int AigNodeIsComplement(int lit) { return lit & 1; }
inline int AigIsNode(int nodeId, int nPIs) {
    return nodeId > nPIs;
} // considering const 1
inline int AigIsPIConst(int nodeId, int nPIs) {
    return nodeId <= nPIs;
}
inline int AigNodeLitCond(int nodeId, int complement) {
    return (int)(((unsigned)nodeId << 1) | (unsigned)(complement != 0));
}
inline int AigNodeNotCond(int lit, int complement) {
    return (int)((unsigned)lit ^ (unsigned)(complement != 0));
}
inline int AigNodeNot(int lit) { return lit ^ 1; }
inline int AigNodeIDDebug(int lit, int nPIs, int nPOs) {
    int id = dUtils::AigNodeID(lit);
    return dUtils::AigIsNode(id, nPIs) ? (id + nPOs) : id;
}
inline int TruthWordNum(int nVars) {
    return nVars <= 5 ? 1 : (1 << (nVars - 5));
}
inline int Truth6WordNum(int nVars) {
    return nVars <= 6 ? 1 : (1 << (nVars - 6));
}

// unary thrust functor
template <typename ValueT, typename MaskT>
struct getValueFilteredByMask {
    const MaskT maskTrue;

    getValueFilteredByMask() : maskTrue(1) {}
    getValueFilteredByMask(MaskT _maskTrue) : maskTrue(_maskTrue) {}

    ValueT operator()(const std::tuple<ValueT, MaskT> &elem) const {
        return std::get<1>(elem) == maskTrue ? std::get<0>(elem) : 0;
    }
};

template <typename IdT, typename LevelT>
struct decreaseLevelIds {

    bool operator()(const std::tuple<IdT, LevelT> &e1,
                    const std::tuple<IdT, LevelT> &e2) const {
        return std::get<1>(e1) == std::get<1>(e2)
                   ? (std::get<0>(e1) > std::get<0>(e2))
                   : (std::get<1>(e1) > std::get<1>(e2));
        // return thrust::get<1>(e1) > thrust::get<1>(e2);
    }
};

template <typename IdT, typename LevelT>
struct decreaseLevels {

    bool operator()(const std::tuple<IdT, LevelT> &e1,
                    const std::tuple<IdT, LevelT> &e2) const {
        return std::get<1>(e1) > std::get<1>(e2);
    }
};

template <typename IdT, typename LevelT>
struct decreaseLevelsPerm {

    bool operator()(const std::tuple<IdT, LevelT, int> &e1,
                    const std::tuple<IdT, LevelT, int> &e2) const {
        return std::get<1>(e1) == std::get<1>(e2)
                   ? (std::get<2>(e1) > std::get<2>(e2))
                   : (std::get<1>(e1) > std::get<1>(e2));
    }
};

template <typename T>
struct isMinusOne { 
    
    bool operator()(const T &elem) const {
        return elem == -1;
    }
};

template <typename T>
struct isOne { 
    
    bool operator()(const T &elem) const {
        return elem == 1;
    }
};

template <typename T>
struct isNotOne { 
    
    bool operator()(const T &elem) const {
        return elem != 1;
    }
};

template <typename T, T val>
struct equalsVal {
    
    bool operator()(const T &elem) const {
        return elem == val;
    }
};

template <typename T, T val>
struct notEqualsVal {
    
    bool operator()(const T &elem) const {
        return elem != val;
    }
};

template <typename T, T val>
struct greaterThanVal {
    
    bool operator()(const T &elem) const {
        return elem > val;
    }
};

template <typename T, T val>
struct greaterThanEqualsValInt {
    
    int operator()(const T &elem) const {
        return (elem >= val ? 1 : 0);
    }
};

template <typename T, T val>
struct lessThanVal {
    
    bool operator()(const T &elem) const {
        return elem < val;
    }
};

struct sameNodeID {
    
    bool operator()(const int &lhs, const int &rhs) const {
        return (lhs >> 1) == (rhs >> 1);
    }
};

template <typename T>
struct isNodeLit {
    const int nPIs;

    isNodeLit() = delete;
    isNodeLit(const int _nPIs) : nPIs(_nPIs) {}

    
    bool operator()(const T &elem) const {
        return (elem >> 1) > nPIs;
    }
};

template <typename T>
struct isPIConstLit {
    const int nPIs;

    isPIConstLit() = delete;
    isPIConstLit(const int _nPIs) : nPIs(_nPIs) {}

    
    bool operator()(const T &elem) const {
        return (elem >> 1) <= nPIs;
    }
};

struct getNodeID {

    int operator()(const int elem) const {
        return elem >> 1;
    }
};

const int AigConst1 = 0;
const int AigConst0 = 1;
} // namespace dUtils

