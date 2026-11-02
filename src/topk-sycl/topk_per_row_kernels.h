// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cfloat>
#include <sycl/sycl.hpp>
#include "utils.hpp"
#include "block_load.hpp"
#include "block_store.hpp"
#include "block_scan.hpp"

// Histogram and Counter live in work-group local memory. Pinning
// local_space + work_group lets the NVIDIA backend emit atom.shared
// instead of a generic-space atomic.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::local_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::work_group>
static inline T atomic_fetch_add(T *addr, const T delta)
{
    sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace> ref(*addr);
    return ref.fetch_add(delta);
}

using fp32x1 = __attribute__((__ext_vector_type__(1))) float;
using fp32x2 = __attribute__((__ext_vector_type__(2))) float;
using fp32x4 = __attribute__((__ext_vector_type__(4))) float;
using fp32x8 = __attribute__((__ext_vector_type__(8))) float;

template <int vec>
struct to_vector;

template <>
struct to_vector<1>
{
    using type = fp32x1;
};

template <>
struct to_vector<2>
{
    using type = fp32x2;
};

template <>
struct to_vector<4>
{
    using type = fp32x4;
};
template <>
struct to_vector<8>
{
    using type = fp32x8;
};

// AIR TopK start

using WideT                             = fp32x4;
constexpr int VECTORIZED_READ_SIZE      = 16;
constexpr int VECTORIZED_TAIL_THREADS   = 32;

enum class Phase
{
    Prefill,
    Decode,
};

template <typename IdxT>
struct ComputeOffset
{
    explicit ComputeOffset(IdxT const& cols) : cols_(cols) {}

    IdxT operator()(IdxT const& x) const { return cols_ * x; }

    IdxT cols_;
};

template <int BitsPerPass>
constexpr int calc_num_buckets()
{
    return 1 << BitsPerPass;
}

/**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr IntType ceildiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}

/**
 * @brief Provide an alignment function ie. ceil(a / b) * b
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr IntType alignTo(IntType a, IntType b)
{
    return ceildiv(a, b) * b;
}

template <typename T, int BitsPerPass>
constexpr int calc_num_passes()
{
    return ceildiv<int>(sizeof(T) * 8, BitsPerPass);
}

int round(int num, int round_value)
{
    return ((num - 1) / round_value + 1) * round_value;
}

template <typename T, int BitsPerPass>
constexpr int calc_start_bit(int pass)
{
    int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
    int r         = start_bit < 0 ? 0 : start_bit;
    return r;
}

template <typename T, int BitsPerPass>
constexpr unsigned calc_mask(int pass)
{
    static_assert(BitsPerPass <= 31);
    int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
    return (1 << num_bits) - 1;
}

template <typename T>
typename Traits<T>::UnsignedBits twiddle_in(T key, bool select_min)
{
    auto bits = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    if constexpr(std::is_same_v<T, float>)
    {
        // TODO: hardcoded for select_min is false!
        uint32_t mask = (key < 0) ? 0 : 0x7fffffff;
        return bits ^ mask;
    }
    else
    {
        bits = Traits<T>::TwiddleIn(bits);
        if(!select_min)
        {
            bits = ~bits;
        }
        return bits;
    }
}

template <typename T>
T twiddle_out(typename Traits<T>::UnsignedBits bits, bool select_min)
{
    if(!select_min)
    {
        bits = ~bits;
    }
    bits = Traits<T>::TwiddleOut(bits);
    return reinterpret_cast<T&>(bits);
}

template <typename T, int BitsPerPass>
int calc_bucket(T x, int start_bit, unsigned mask, bool select_min)
{
    static_assert(BitsPerPass <= sizeof(int) * 8 - 1,
                  "BitsPerPass is too large that the result type could not be int");
    return (twiddle_in(x, select_min) >> start_bit) & mask;
}

template <typename I>
constexpr inline std::enable_if_t<std::is_integral<I>::value, bool>
is_a_power_of_two(I val) noexcept
{
    return ((val - 1) & val) == 0;
}

template <typename T, typename IdxT, typename RATIO_T = float>
IdxT calc_buf_len(IdxT len)
{
    // When writing is skipped, only read `in`(type T).
    // When writing is not skipped, read `in_buf`(T) and `in_idx_buf`(IdxT), and
    // write `out_buf`(T) and `out_idx_buf`(IdxT). The ratio between these cases
    // determines whether to skip writing and hence the buffer size.
    constexpr RATIO_T ratio = 2 + sizeof(IdxT) * 2 / sizeof(T);
    // Even such estimation is too conservative, so further decrease buf_len by
    // 1/8
    IdxT buf_len = len / (ratio * 8);

    // one-block kernel splits one large buffer into smaller ones, so round buf
    // size to 256 bytes to avoid alignment issues
    static_assert(is_a_power_of_two(sizeof(T)));
    static_assert(is_a_power_of_two(sizeof(IdxT)));
    constexpr IdxT aligned = 256 / std::min(sizeof(T), sizeof(IdxT));
    buf_len                = buf_len & (~(aligned - 1));
    return buf_len;
}

/**
 * Map a Func over the input data, using vectorized load instructions if
 * possible.
 *
 * NB: in future, we should move this to
 * cpp/include/raft/linalg/detail/unary_op.cuh, which currently does not support
 * the second lambda argument (index of an element)
 *
 * @tparam T element type
 * @tparam IdxT indexing type
 * @tparam Func void (T x, IdxT idx)
 *
 * @param thread_rank rank of the calling thread among all participating threads
 * @param num_threads number of the threads that participate in processing
 * @param in the input data
 * @param len the number of elements to read
 * @param f the lambda taking two arguments (T x, IdxT idx)
 */
template <typename T, typename IdxT, typename Func>
void
vectorized_process(size_t thread_rank, size_t num_threads, T const* in, IdxT len, Func f)
{
    T val;
    int acc          = 0;
    int prev_bin_idx = -1;

    if constexpr(sizeof(T) >= sizeof(WideT))
    {
        for(IdxT i = thread_rank; i < len; i += num_threads)
        {
            val = in[i];
            f(in[i], i, acc, prev_bin_idx, false);
        }
    }
    else
    {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);

        // TODO: it's UB
        union
        {
            WideT scalar;
            T array[items_per_scalar];
        } wide;

        int skip_cnt =
            (reinterpret_cast<size_t>(in) % sizeof(WideT))
                ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
                : 0;
        if(skip_cnt > len)
        {
            skip_cnt = len;
        }
        WideT const* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
        const IdxT len_cast  = (len - skip_cnt) / items_per_scalar;

        for(IdxT i = thread_rank; i < len_cast; i += num_threads)
        {
            wide.scalar       = in_cast[i];
            const IdxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
            for(int j = 0; j < items_per_scalar; ++j)
            {
                val = wide.array[j];
                f(wide.array[j], real_i + j, acc, prev_bin_idx, false);
            }
        }

        static_assert(VECTORIZED_TAIL_THREADS >= items_per_scalar);
        // and because items_per_scalar > skip_cnt, VECTORIZED_TAIL_THREADS > skip_cnt
        // no need to use loop
        if(thread_rank < skip_cnt)
        {
            val = in[thread_rank];
            f(in[thread_rank], thread_rank, acc, prev_bin_idx, false);
        }
        // because len_cast = (len - skip_cnt) / items_per_scalar,
        // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
        // and so
        // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
        // VECTORIZED_TAIL_THREADS, so no loop is needed
        const IdxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
        if(remain_i < len)
        {
            val = in[remain_i];
            f(in[remain_i], remain_i, acc, prev_bin_idx, false);
        }
    }

    if(acc > 0)
    {
        f(-val, 0, acc, prev_bin_idx, true);
    }
}

template <typename T, typename IdxT>
struct alignas(128) Counter
{
    // We are processing the values in multiple passes, from most significant to
    // least significant. In each pass, we keep the length of input (`len`) and
    // the `k` of current pass, and update them at the end of the pass.
    IdxT k;
    IdxT len;

    //  `previous_len` is the length of input in previous pass. Note that
    //  `previous_len` rather than `len` is used for the filtering step because
    //  filtering is indeed for previous pass (see comments before
    //  `radix_kernel`).
    IdxT previous_len;

    // We determine the bits of the k_th value inside the mask processed by the
    // pass. The already known bits are stored in `kth_value_bits`. It's used to
    // discriminate a element is a result (written to `out`), a candidate for next
    // pass (written to `out_buf`), or not useful (discarded). The bits that are
    // not yet processed do not matter for this purpose.
    typename Traits<T>::UnsignedBits kth_value_bits;

    // Record how many elements have passed filtering. It's used to determine the
    // position in the `out_buf` where an element should be written.
    alignas(128) IdxT filter_cnt;

    // For a row inside a batch, we may launch multiple thread blocks. This
    // counter is used to determine if the current block is the last running
    // block. If so, this block will execute scan() and choose_bucket().
    alignas(128) unsigned int finished_block_cnt;

    // Record how many elements have been written to the front of `out`. Elements
    // less (if select_min==true) than the k-th value are written from front to
    // back.
    alignas(128) IdxT out_cnt;

    // Record how many elements have been written to the back of `out`. Elements
    // equal to the k-th value are written from back to front. We need to keep
    // count of them separately because the number of elements that <= the k-th
    // value might exceed k.
    alignas(128) IdxT out_back_cnt;
};

/**
 * Replace histogram with its own prefix sum
 * (step 2 in `radix_kernel` description)
 */
template <typename IdxT, int BitsPerPass, int BlockSize>
void scan(sycl::nd_item<3> &item, IdxT volatile* histogram)
{
    auto g = item.get_group();
    
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    if constexpr(num_buckets >= BlockSize)
    {
        static_assert(num_buckets % BlockSize == 0);
        constexpr int items_per_thread = num_buckets / BlockSize;
         
        typedef BlockLoad<IdxT, BlockSize, items_per_thread, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
        typedef BlockStore<IdxT, BlockSize, items_per_thread, BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
        typedef BlockScan<IdxT, BlockSize> BlockScanT;

        // Shared memory
        union temp_storage
        {
          typename BlockLoadT::TempStorage  load;
          typename BlockStoreT::TempStorage store;
          typename BlockScanT::TempStorage  scan;
        } ;

        auto &slm = *sycl::ext::oneapi::group_local_memory_for_overwrite<union temp_storage>(g);

        IdxT thread_data[items_per_thread];

        BlockLoadT(slm.load, item).Load(histogram, thread_data);
        item.barrier(sycl::access::fence_space::local_space);

        BlockScanT(slm.scan, item).InclusiveSum(thread_data, thread_data);

        item.barrier(sycl::access::fence_space::local_space);

        BlockStoreT(slm.store, item).Store(histogram, thread_data);
    }
    else
    {
        IdxT thread_data = 0;
        if (item.get_local_id(2) < num_buckets)
        {
            thread_data = histogram[item.get_local_id(2)];
        }

        typedef BlockScan<IdxT, BlockSize> BlockScanT;
        typename BlockScanT::TempStorage scan;
        auto &slm = *sycl::ext::oneapi::group_local_memory_for_overwrite<scan>(g);
        BlockScan(slm, item).InclusiveSum(thread_data, thread_data);

        item.barrier(sycl::access::fence_space::local_space);

        if (item.get_local_id(2) < num_buckets)
        {
            histogram[item.get_local_id(2)] = thread_data;
        }
    }
}

/**
 * Calculate in which bucket the k-th value will fall
 *  (steps 3 in `radix_kernel` description)
 */
template <typename T, typename IdxT, int BitsPerPass>
void
choose_bucket(sycl::nd_item<3> &item, Counter<T, IdxT>* counter,
              IdxT const* histogram, const IdxT k, int const pass)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    for (int i = item.get_local_id(2); i < num_buckets;
         i += item.get_local_range(2))
    {
        IdxT prev = (i == 0) ? 0 : histogram[i - 1];
        IdxT cur  = histogram[i];

        // one and only one thread will satisfy this condition, so counter is
        // written by only one thread
        if(prev < k && cur >= k)
        {
            counter->k   = k - prev;   // how many values still are there to find
            counter->len = cur - prev; // number of values in next pass
            typename Traits<T>::UnsignedBits bucket = i;
            int start_bit                                   = calc_start_bit<T, BitsPerPass>(pass);
            counter->kth_value_bits |= bucket << start_bit;
        }
    }
}

// For one-block version, last_filter() could be called when pass < num_passes
// - 1. So `pass` could not be constexpr
template <typename T,
          typename IdxT,
          int BitsPerPass,
          bool WRITE_TOPK_VALUES,
          bool prioritize_smaller_indice = false>
void last_filter(sycl::nd_item<3> &item,
                 T const* in_buf,
                 IdxT const* in_idx_buf,
                 T* out,
                 IdxT* out_idx,
                 IdxT current_len,
                 IdxT k,
                 Counter<T, IdxT>* counter,
                 bool const select_min,
                 int const pass,
                 bool const use_one_pass = false)
{
    auto const kth_value_bits = counter->kth_value_bits;
    int const start_bit       = calc_start_bit<T, BitsPerPass>(pass);

    // changed in choose_bucket(); need to reload
    const IdxT num_of_kth_needed = counter->k;
    IdxT *p_out_cnt              = &counter->out_cnt;
    IdxT *p_out_back_cnt         = &counter->out_back_cnt;
    //IdxT* p_equal                = out_idx + k - num_of_kth_needed;
    if(in_idx_buf)
    {
        for (IdxT i = item.get_local_id(2); i < current_len;
             i += item.get_local_range(2))
        {
            const T value   = in_buf[i];
            auto const bits = use_one_pass
                                  ? twiddle_in(value, select_min) & ((1 << BitsPerPass) - 1)
                                  : (twiddle_in(value, select_min) >> start_bit) << start_bit;
            if(bits < kth_value_bits)
            {
                IdxT pos = atomic_fetch_add(p_out_cnt, static_cast<IdxT>(1));
                if(WRITE_TOPK_VALUES)
                {
                    out[pos] = value;
                }
                // For one-block version, `in_idx_buf` could be nullptr at pass 0.
                // For non one-block version, if writing has been skipped, `in_idx_buf`
                // could be nullptr if `in_buf` is `in`
                out_idx[pos] = in_idx_buf[i];
            }
            else if(bits == kth_value_bits)
            {
                IdxT new_idx  = in_idx_buf[i];
                IdxT back_pos = atomic_fetch_add(p_out_back_cnt, static_cast<IdxT>(1));
                if(back_pos < num_of_kth_needed)
                {
                    IdxT pos = k - 1 - back_pos;
                    if(WRITE_TOPK_VALUES)
                    {
                        out[pos] = value;
                    }
                    if constexpr(!prioritize_smaller_indice)
                    {
                        out_idx[pos] = new_idx;
                    }
                }
            }
        }
    }
    else
    {
        for (IdxT i = item.get_local_id(2); i < current_len;
             i += item.get_local_range(2))
        {
            const T value   = in_buf[i];
            auto const bits = use_one_pass
                                  ? twiddle_in(value, select_min) & ((1 << BitsPerPass) - 1)
                                  : (twiddle_in(value, select_min) >> start_bit) << start_bit;
            if(bits < kth_value_bits)
            {
                IdxT pos = atomic_fetch_add(p_out_cnt, static_cast<IdxT>(1));
                if(WRITE_TOPK_VALUES)
                {
                    out[pos] = value;
                }
                // For one-block version, `in_idx_buf` could be nullptr at pass 0.
                // For non one-block version, if writing has been skipped, `in_idx_buf`
                // could be nullptr if `in_buf` is `in`
                out_idx[pos] = i;
            }
            else if(bits == kth_value_bits)
            {
                IdxT new_idx  = i;
                IdxT back_pos = atomic_fetch_add(p_out_back_cnt, static_cast<IdxT>(1));
                if(back_pos < num_of_kth_needed)
                {
                    IdxT pos = k - 1 - back_pos;
                    if(WRITE_TOPK_VALUES)
                    {
                        out[pos] = value;
                    }
                    if constexpr(!prioritize_smaller_indice)
                    {
                        out_idx[pos] = new_idx;
                    }
                }
            }
        }
    }
}

template <typename T, typename IdxT>
void set_buf_pointers(T const* in,
                      IdxT const* in_idx,
                      T* buf1,
                      IdxT* idx_buf1,
                      T* buf2,
                      IdxT* idx_buf2,
                      int pass,
                      T const*& in_buf,
                      IdxT const*& in_idx_buf,
                      T*& out_buf,
                      IdxT*& out_idx_buf)
{
    if(pass == 0)
    {
        in_buf      = in;
        in_idx_buf  = nullptr;
        out_buf     = nullptr;
        out_idx_buf = nullptr;
    }
    else if(pass == 1)
    {
        in_buf      = in;
        in_idx_buf  = in_idx;
        out_buf     = buf1;
        out_idx_buf = idx_buf1;
    }
    else if(pass % 2 == 0)
    {
        in_buf      = buf1;
        in_idx_buf  = idx_buf1;
        out_buf     = buf2;
        out_idx_buf = idx_buf2;
    }
    else
    {
        in_buf      = buf2;
        in_idx_buf  = idx_buf2;
        out_buf     = buf1;
        out_idx_buf = idx_buf1;
    }
}

template <typename T, typename IdxT>
void set_buf_pointers(T const* in,
                                 IdxT const* in_idx,
                                 char* bufs,
                                 IdxT buf_len,
                                 int pass,
                                 T const*& in_buf,
                                 IdxT const*& in_idx_buf,
                                 T*& out_buf,
                                 IdxT*& out_idx_buf)
{
    // bufs consists of 4 pieces in order: buf1, buf2, idx_buf1, idx_buf2
    if(pass == 0)
    {
        in_buf      = in;
        in_idx_buf  = nullptr;
        out_buf     = nullptr;
        out_idx_buf = nullptr;
    }
    else if(pass == 1)
    {
        in_buf      = in;
        in_idx_buf  = in_idx;
        out_buf     = reinterpret_cast<T*>(bufs);
        out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
    }
    else if(pass % 2 == 0)
    {
        in_buf      = reinterpret_cast<T*>(bufs);
        in_idx_buf  = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
        out_buf     = const_cast<T*>(in_buf + buf_len);
        out_idx_buf = const_cast<IdxT*>(in_idx_buf + buf_len);
    }
    else
    {
        out_buf     = reinterpret_cast<T*>(bufs);
        out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
        in_buf      = out_buf + buf_len;
        in_idx_buf  = out_idx_buf + buf_len;
    }
}

// The following a few functions are for the one-block version, which uses
// single thread block for each row of a batch.
template <typename T, typename IdxT, int BitsPerPass, bool WRITE_TOPK_VALUES,
          int BlockSize>
bool filter_and_histogram_for_one_block(sycl::nd_item<3> &item,
                                        T const *in_buf, IdxT const *in_idx_buf,
                                        T *out_buf, IdxT *out_idx_buf, T *out,
                                        IdxT *out_idx, const IdxT previous_len,
                                        Counter<T, IdxT> *counter,
                                        IdxT *histogram, bool select_min,
                                        int pass, IdxT k, bool &use_one_pass_sm)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    for (int i = item.get_local_id(2); i < num_buckets * 2;
         i += item.get_local_range(2))
    {
        histogram[i] = 0;
    }
    IdxT *p_filter_cnt = &counter->filter_cnt;
    if (item.get_local_id(2) == 0)
    {
        *p_filter_cnt = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    int const start_bit = calc_start_bit<T, BitsPerPass>(pass);
    unsigned const mask = calc_mask<T, BitsPerPass>(pass);

    if(pass == 0)
    {
        T local_min = std::numeric_limits<T>::max();
        T local_max = std::numeric_limits<T>::lowest();

        auto f = [histogram, select_min, start_bit, mask, &local_min, &local_max](
                     T value, IdxT, int& acc, int& prev_bin_idx, bool is_last) {
            int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
            // atomicAdd(histogram + bucket, static_cast<IdxT>(1));

            if(bucket == prev_bin_idx)
            {
                acc++;
            }
            else
            {
                if(acc > 0)
                {
                    atomic_fetch_add(histogram + prev_bin_idx, static_cast<IdxT>(acc));
                }
                acc          = 1;
                prev_bin_idx = bucket;
            }

            if(is_last)
            {
                return;
            }

            int bucket_low =
                calc_bucket<T, BitsPerPass>(value, 0, (1 << BitsPerPass) - 1, select_min);
            atomic_fetch_add(histogram + num_buckets + bucket_low, static_cast<IdxT>(1));

            local_min = sycl::fmin(local_min, value);
            local_max = sycl::fmax(local_max, value);
        };
        vectorized_process(item.get_local_id(2),
                           item.get_local_range(2), in_buf, previous_len,
                           f);

        T global_min = sycl::reduce_over_group(item.get_group(), local_min, sycl::minimum<>());
        T global_max = sycl::reduce_over_group(item.get_group(), local_max, sycl::maximum<>());

        if (item.get_local_id(2) == 0)
        {
            auto global_min_bits = twiddle_in(global_min, select_min);
            auto global_max_bits = twiddle_in(global_max, select_min);
            uint32_t diff        = global_min_bits ^ global_max_bits;
            use_one_pass_sm      = diff < (1u << BitsPerPass);
        }
        item.barrier(sycl::access::fence_space::local_space);

        return use_one_pass_sm;
    }
    else if(!out_buf)
    {
        // not use vectorized_process here because it increases #registers a lot
        auto const kth_value_bits    = counter->kth_value_bits;
        int const previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

        for (IdxT i = item.get_local_id(2); i < previous_len;
             i += item.get_local_range(2))
        {
            const T value            = in_buf[i];
            auto const previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                       << previous_start_bit;
            if(previous_bits == kth_value_bits)
            {
                int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
                atomic_fetch_add(histogram + bucket, static_cast<IdxT>(1));
            }
        }
    }
    else
    {
        // not use vectorized_process here because it increases #registers a lot
        IdxT *p_out_cnt              = &counter->out_cnt;
        auto const kth_value_bits    = counter->kth_value_bits;
        int const previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

        if(in_idx_buf)
        {
            for (IdxT i = item.get_local_id(2); i < previous_len;
                 i += item.get_local_range(2))
            {
                const T value            = in_buf[i];
                auto const previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                           << previous_start_bit;
                if(previous_bits == kth_value_bits)
                {

                    IdxT pos = atomic_fetch_add(p_filter_cnt, static_cast<IdxT>(1));
                    out_buf[pos]     = value;
                    out_idx_buf[pos] = in_idx_buf[i];

                    int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
                    atomic_fetch_add(histogram + bucket, static_cast<IdxT>(1));
                }
                else if(previous_bits < kth_value_bits)
                {
                    IdxT pos = atomic_fetch_add(p_out_cnt, static_cast<IdxT>(1));
                    if(WRITE_TOPK_VALUES)
                    {
                        out[pos] = value;
                    }
                    out_idx[pos] = in_idx_buf[i];
                }
            }
        }
        else
        {
            for (IdxT i = item.get_local_id(2); i < previous_len;
                 i += item.get_local_range(2))
            {
                const T value            = in_buf[i];
                auto const previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                           << previous_start_bit;
                if(previous_bits == kth_value_bits)
                {

                    IdxT pos = atomic_fetch_add(p_filter_cnt, static_cast<IdxT>(1));
                    out_buf[pos]     = value;
                    out_idx_buf[pos] = i;

                    int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
                    atomic_fetch_add(histogram + bucket, static_cast<IdxT>(1));
                }
                else if(previous_bits < kth_value_bits)
                {
                    IdxT pos = atomic_fetch_add(p_out_cnt, static_cast<IdxT>(1));
                    if(WRITE_TOPK_VALUES)
                    {
                        out[pos] = value;
                    }
                    out_idx[pos] = i;
                }
            }
        }
    }

    return false;
}

template <typename T,
          typename IdxT,
          int BitsPerPass,
          int BlockSize,
          bool WRITE_TOPK_VALUES,
          bool prioritize_smaller_indice = false,
          Phase phase>
void radix_topk_one_block_kernel(sycl::nd_item<3> &item,
                                 T const* in,
                                 IdxT const* in_idx,
                                 const int64_t len,
                                 const IdxT* rowStarts,
                                 const IdxT* rowEnds,
                                 const IdxT k,
                                 T* out,
                                 IdxT* out_idx,
                                 bool const select_min,
                                 char* bufs,
                                 const int next_n,
                                 bool &use_one_pass_sm,
                                 Counter<T, IdxT> &counter,
                                 IdxT *histogram)
{
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();

    const int64_t batch_id = item.get_group(2);

    IdxT rowStart = 0;
    IdxT rowEnd   = len;
    if(phase == Phase::Prefill)
    {
        if(rowStarts && rowEnds)
        {
            rowStart = rowStarts[batch_id];
            rowEnd   = rowEnds[batch_id];
        }
    }
    else
    {
        rowEnd   = rowEnds[batch_id / next_n] - next_n + (batch_id % next_n) + 1;
        rowStart = 0;
    }

    const IdxT row_len = rowEnd - rowStart;

    if (item.get_local_id(2) == 0)
    {
        counter.k              = k;
        counter.len            = row_len;
        counter.previous_len   = row_len;
        counter.kth_value_bits = 0;
        counter.out_cnt        = 0;
        counter.out_back_cnt   = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    in += batch_id * len;
    out += batch_id * k;
    out_idx += batch_id * k;
    if(in_idx)
    {
        in_idx += batch_id * len;
    }

    if(row_len <= k)
    {
        for (int rowIt = item.get_local_id(2); rowIt < k;
             rowIt += BlockSize)
        {
            out_idx[rowIt] = rowIt < row_len ? rowIt + rowStart : -1;
            if(WRITE_TOPK_VALUES)
            {
                out[rowIt] = rowIt < row_len ? in[rowIt + rowStart] : 0;
            }
        }
        return;
    }

    const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);
    bufs += batch_id * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
    for(int pass = 0; pass < num_passes; ++pass)
    {
        T const* in_buf        = nullptr;
        IdxT const* in_idx_buf = nullptr;
        T* out_buf             = nullptr;
        IdxT* out_idx_buf      = nullptr;
        set_buf_pointers(in, in_idx, bufs, buf_len, pass, in_buf, in_idx_buf, out_buf, out_idx_buf);

        const IdxT current_len = counter.len;
        const IdxT current_k   = counter.k;
        IdxT previous_len      = counter.previous_len;
        if(previous_len > buf_len)
        {
            in_buf       = in;
            in_idx_buf   = in_idx;
            previous_len = row_len;
        }
        if(current_len > buf_len)
        {
            // so "out_buf==nullptr" denotes skipping writing buffer in current pass
            out_buf     = nullptr;
            out_idx_buf = nullptr;
        }

        const bool use_one_pass =
            filter_and_histogram_for_one_block<T, IdxT, BitsPerPass,
                                               WRITE_TOPK_VALUES, BlockSize>(
                item,
                in_buf, in_idx_buf, out_buf, out_idx_buf, out, out_idx,
                previous_len, &counter, histogram, select_min, pass, k,
                use_one_pass_sm); //@TODO CHECK UPDATE CODE
        item.barrier(sycl::access::fence_space::local_space);

        scan<IdxT, BitsPerPass, BlockSize>(item, histogram + use_one_pass * num_buckets);
        item.barrier(sycl::access::fence_space::local_space);

        choose_bucket<T, IdxT, BitsPerPass>(item,
                                            &counter,
                                            histogram + use_one_pass * num_buckets,
                                            current_k,
                                            pass + use_one_pass * num_passes);
        if (item.get_local_id(2) == 0)
        {
            counter.previous_len = current_len;
        }
        item.barrier(sycl::access::fence_space::local_space);

        if(use_one_pass || pass == num_passes - 1)
        {
            last_filter<T, IdxT, BitsPerPass, WRITE_TOPK_VALUES, prioritize_smaller_indice>(
                item,
                out_buf ? out_buf : in,
                out_buf ? out_idx_buf : in_idx,
                out,
                out_idx,
                out_buf ? current_len : row_len,
                k,
                &counter,
                select_min,
                pass,
                use_one_pass);
            break;
        }
        else if(counter.len == counter.k)
        {
            last_filter<T, IdxT, BitsPerPass, WRITE_TOPK_VALUES, false>(
                item,
                out_buf ? out_buf : in,
                out_buf ? out_idx_buf : in_idx,
                out,
                out_idx,
                out_buf ? current_len : row_len,
                k,
                &counter,
                select_min,
                pass);
            break;
        }
    }
}

inline size_t calc_aligned_size(std::vector<size_t> const& sizes)
{
    const size_t ALIGN_BYTES = 256;
    const size_t ALIGN_MASK  = ~(ALIGN_BYTES - 1);
    size_t total             = 0;
    for(auto sz : sizes)
    {
        total += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
    }
    return total + ALIGN_BYTES - 1;
}

inline std::vector<void*> calc_aligned_pointers(void const* p, std::vector<size_t> const& sizes)
{
    const size_t ALIGN_BYTES = 256;
    const size_t ALIGN_MASK  = ~(ALIGN_BYTES - 1);

    char* ptr =
        reinterpret_cast<char*>((reinterpret_cast<size_t>(p) + ALIGN_BYTES - 1) & ALIGN_MASK);

    std::vector<void*> aligned_pointers;
    aligned_pointers.reserve(sizes.size());
    for(auto sz : sizes)
    {
        aligned_pointers.push_back(ptr);
        ptr += (sz + ALIGN_BYTES - 1) & ALIGN_MASK;
    }

    return aligned_pointers;
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize,
          bool WRITE_TOPK_VALUES, Phase phase = Phase::Prefill>
void standalone_stable_radix_topk_one_block_(
    void *buf, size_t &buf_size, T const *in, IdxT const *in_idx,
    int batch_size, int64_t len, IdxT *rowStarts, IdxT *rowEnds, IdxT k, T *out,
    IdxT *out_idx, bool select_min, sycl::queue &q, bool sorted = false,
    int next_n = 0)
{
    static_assert(calc_num_passes<T, BitsPerPass>() > 1);

    char* bufs         = nullptr;
    const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);

    {
        size_t total_size         = 0;
        std::vector<size_t> sizes = {buf_len * 2 * (sizeof(T) + sizeof(IdxT)) * batch_size};

        total_size = calc_aligned_size(sizes);

        if(!buf)
        {
            buf_size = total_size;
            return;
        }

        std::vector<void*> aligned_pointers = calc_aligned_pointers(buf, sizes);
        bufs                                = static_cast<decltype(bufs)>(aligned_pointers[0]);
    }

    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();

    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<bool, 0> use_one_pass_sm(cgh);
        sycl::local_accessor<Counter<T, IdxT>, 0> counter_acc(cgh);
        sycl::local_accessor<IdxT, 1> histogram_acc(sycl::range<1>(num_buckets * 2), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size * BlockSize),
                              sycl::range<3>(1, 1, BlockSize)),
            [=](sycl::nd_item<3> item) {
                radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize,
                                            WRITE_TOPK_VALUES, false, phase>(
                    item,
                    in, in_idx, len, rowStarts, rowEnds, k, out, out_idx,
                    select_min, bufs, next_n, use_one_pass_sm, counter_acc,
                    histogram_acc.template get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

template <typename T, typename IdxT, bool WRITE_TOPK_VALUES,
          bool sorted = false, Phase phase = Phase::Prefill>
void standalone_stable_radix_11bits(void *buf, size_t &buf_size, T const *in,
                                    int batch_size, int64_t len,
                                    IdxT *rowStarts, IdxT *rowEnds, IdxT k,
                                    T *out, IdxT *out_idx, bool greater,
                                    sycl::queue &q, int next_n = 0)
{
    constexpr int items_per_thread   = 32;
    constexpr int block_dim          = 1024;
    if(len <= block_dim * items_per_thread)
    {
        standalone_stable_radix_topk_one_block_<T, IdxT, 11, block_dim, WRITE_TOPK_VALUES, phase>(
            buf,
            buf_size,
            in,
            static_cast<IdxT*>(nullptr),
            batch_size,
            len,
            rowStarts,
            rowEnds,
            k,
            out,
            out_idx,
            !greater,
            q,
            sorted,
            next_n);
    }
    else
    {
        standalone_stable_radix_topk_one_block_<T,
                                                IdxT,
                                                11,
                                                block_dim,
                                                WRITE_TOPK_VALUES,
                                                phase>(buf,
                                                       buf_size,
                                                       in,
                                                       static_cast<IdxT*>(nullptr),
                                                       batch_size,
                                                       len,
                                                       rowStarts,
                                                       rowEnds,
                                                       k,
                                                       out,
                                                       out_idx,
                                                       !greater,
                                                       q,
                                                       sorted,
                                                       next_n);
    }
}

// Explicit template instantiation for standalone_stable_radix_11bits
template void standalone_stable_radix_11bits<float, int, true, true>(
    void *buf, size_t &buf_size, float const *in, int batch_size, int64_t len,
    int *rowStarts, int *rowEnds, int k, float *out, int *out_idx, bool greater,
    sycl::queue &q, int next_n);

template void standalone_stable_radix_11bits<float, int, false, true>(
    void *buf, size_t &buf_size, float const *in, int batch_size, int64_t len,
    int *rowStarts, int *rowEnds, int k, float *out, int *out_idx, bool greater,
    sycl::queue &q, int next_n);

template <typename T, Phase phase = Phase::Prefill>
int64_t invokeComputeTopkLastDimWorkspaceSize(int32_t numRows, int32_t stride0, sycl::queue &q)
{
    using IdxT = int32_t;

    size_t buf_size = 0;
    void* workspace = nullptr;
    T const* in     = nullptr;
    T* out_val      = nullptr;
    IdxT* out_idx   = nullptr;

    constexpr int block_dim          = 1024;
    constexpr bool sorted            = true;
    constexpr bool is_largest        = true;
    constexpr int k                  = 2048;

    standalone_stable_radix_topk_one_block_<T, IdxT, 11, block_dim, false>(
        workspace, buf_size, in, static_cast<IdxT *>(nullptr), numRows, stride0,
        static_cast<IdxT *>(nullptr), static_cast<IdxT *>(nullptr), k, out_val,
        out_idx, !is_largest, q, sorted);
    return buf_size;
}

// Explicit template instantiation to ensure the symbol is available for linking
template int64_t invokeComputeTopkLastDimWorkspaceSize<float>(int32_t numRows, int32_t stride0, sycl::queue &q);

// Helper function to determine if topk_per_row kernel should be used
// Based on: n + K log²K ≥ 3 × Factor(n) × n
// where Factor(n) = 1/3 + 1.6/(log₂(n) - 9.5)
// Simplifies to: K log²K ≥ 4.8n/(log₂(n) - 9.5)
// TODO: We need to confirm whether, when n <= 2048, we might choose
// radix sort because the denominator becomes very small; does that
// still yield the best performance?
constexpr int MAX_CAPACITY = 2048;

template <typename IdxT>
inline bool should_use_topk_radix(IdxT len, IdxT k)
{
    const double n = static_cast<double>(len);
    const double K = static_cast<double>(k);

    if(K <= 1.0)
    {
        return false;
    }

    const double log_n = std::log2(n);

    const double denom = std::max(0.0001, log_n - 9.5);

    const double rhs = (4.8 * n) / denom;

    const double log_k = std::log2(K);
    const double lhs   = K * log_k * log_k;

    return lhs >= rhs;
}

// Helper function to call topk_per_row kernel
template <typename IdxT>
void topk_per_row_kernel_launcher(const float *in, const IdxT *rowStarts,
                                  const IdxT *rowEnds, IdxT *out_idx,
                                  const float *out, int batch_size, int stride0,
                                  int stride1, int k,
                                  sycl::queue &q) try {

    size_t buf_size = 0; // will be overwritten by the kernel

    static constexpr bool is_largest = true;

    int64_t workspace_size = invokeComputeTopkLastDimWorkspaceSize<float>(batch_size, stride0, q);

    uint8_t *workspace = (uint8_t *)sycl::malloc_device(workspace_size, q);

    if(out)
    {
        standalone_stable_radix_11bits<float, int, true, true>(
            workspace,
            buf_size,
            in,
            batch_size,
            stride0,
            const_cast<IdxT*>(rowStarts),
            const_cast<IdxT*>(rowEnds),
            k,
            const_cast<float*>(out),
            out_idx,
            is_largest,
            q);
    }
    else
    {
        standalone_stable_radix_11bits<float, int, false, true>(
            workspace,
            buf_size,
            in,
            batch_size,
            stride0,
            const_cast<IdxT*>(rowStarts),
            const_cast<IdxT*>(rowEnds),
            k,
            nullptr,
            out_idx,
            is_largest,
            q);
    }

    q.wait();
    sycl::free(workspace, q);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <bool greater, typename T, typename IdxT>
void AdaptiveTopK(int batch_size, IdxT len, IdxT k, const T *__restrict__ in,
                  T *__restrict__ out, IdxT *__restrict__ out_idx,
                  sycl::queue &q)
{
    assert(k <= MAX_CAPACITY);

    constexpr bool is_float = std::is_same_v<T, float>;
    if constexpr(is_float)
    {
        // Use topk_per_row kernel when:
        // n + K log²K ≥ 3 × Factor(n) × n
        // where Factor(n) = 1/3 + 1.6/(log₂(n) - 9.5)
        assert(should_use_topk_radix(len, k) && greater);

        topk_per_row_kernel_launcher<IdxT>(in,
                                           nullptr,
                                           nullptr,
                                           out_idx,
                                           out,
                                           batch_size,
                                           static_cast<int>(len),
                                           1,
                                           k,
                                           q);
    }
}

template <typename T, typename IdxT>
void topk_radix(T *values,      // [batch, len]
                IdxT *topk_ids, // [batch, k]
                T *topk_out,    // [batch, k]
                int topk, bool largest, int32_t *rowStarts, int32_t *rowEnds,
                int64_t stride0, int64_t stride1, int32_t batch,
                int32_t max_len,
                sycl::queue &q)
{
    // not using variable length mode
    //const bool use_variable_length = rowStarts != nullptr && rowEnds != nullptr;

    // Set default stride values if not specified
    if(stride0 < 0) stride0 = max_len;

    AdaptiveTopK<true, T, IdxT>(batch, max_len, topk, values, topk_out, topk_ids, q);
}
