/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * WarpScanShfl provides SHFL-based variants of parallel prefix scan of items partitioned across a CUDA thread warp.
 */

#pragma once

#if defined(__SYCL_DEVICE_ONLY__) && defined(__INTEL_LLVM_COMPILER)
template <typename T>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT __attribute__((noduplicate))
T __spirv_GroupNonUniformShuffle(__spv::Scope::Flag, T, unsigned) noexcept;

template <typename T>
__SYCL_CONVERGENT__ extern SYCL_EXTERNAL __SYCL_EXPORT __attribute__((noduplicate))
T __spirv_GroupNonUniformShuffleUp(__spv::Scope::Flag, T, unsigned) noexcept;
#endif


#if defined(__NVPTX__) && defined(__SYCL_DEVICE_ONLY__)
#define SHFL_SYNC(RES, MASK, VAL, SHFL_PARAM, C, SHUFFLE_INSTR)                \
  if constexpr (std::is_same_v<T, double>) {                                   \
    int x_a, x_b;                                                              \
    asm("mov.b64 {%0,%1},%2;" : "=r"(x_a), "=r"(x_b) : "d"(VAL));              \
    auto tmp_a = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_a, SHFL_PARAM, C);   \
    auto tmp_b = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_b, SHFL_PARAM, C);   \
    asm("mov.b64 %0,{%1,%2};" : "=d"(RES) : "r"(tmp_a), "r"(tmp_b));           \
  } else if constexpr (std::is_same_v<T, long> ||                              \
                       std::is_same_v<T, unsigned long>) {                     \
    int x_a, x_b;                                                              \
    asm("mov.b64 {%0,%1},%2;" : "=r"(x_a), "=r"(x_b) : "l"(VAL));              \
    auto tmp_a = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_a, SHFL_PARAM, C);   \
    auto tmp_b = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_b, SHFL_PARAM, C);   \
    asm("mov.b64 %0,{%1,%2};" : "=l"(RES) : "r"(tmp_a), "r"(tmp_b));           \
  } else if constexpr (std::is_same_v<T, sycl::half>) {                        \
    short tmp_b16;                                                             \
    asm("mov.b16 %0,%1;" : "=h"(tmp_b16) : "h"(VAL));                          \
    auto tmp_b32 = __nvvm_shfl_sync_##SHUFFLE_INSTR(                           \
        MASK, static_cast<int>(tmp_b16), SHFL_PARAM, C);                       \
    asm("mov.b16 %0,%1;" : "=h"(RES) : "h"(static_cast<short>(tmp_b32)));      \
  } else if constexpr (std::is_same_v<T, float>) {                             \
    auto tmp_b32 = __nvvm_shfl_sync_##SHUFFLE_INSTR(                           \
        MASK, __nvvm_bitcast_f2i(VAL), SHFL_PARAM, C);                         \
    RES = __nvvm_bitcast_i2f(tmp_b32);                                         \
  } else {                                                                     \
    RES = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, VAL, SHFL_PARAM, C);          \
  }
#endif

template <typename T>
T select_from_sub_group(unsigned int member_mask,
                        sycl::sub_group g, T x, int remote_local_id,
                        int logical_sub_group_size = 32) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__SPIR__)
  unsigned int start_index =
      g.get_local_linear_id() / logical_sub_group_size * logical_sub_group_size;
  unsigned logical_remote_id =
      start_index + remote_local_id % logical_sub_group_size;
  return __spirv_GroupNonUniformShuffle(__spv::Scope::Subgroup, x, logical_remote_id);
#elif defined(__NVPTX__)
  T result;
  int cVal = ((32 - logical_sub_group_size) << 8) | 31;
  SHFL_SYNC(result, member_mask, x, remote_local_id, cVal, idx_i32)
  return result;
#endif
#else
  (void)g;
  (void)x;
  (void)remote_local_id;
  (void)logical_sub_group_size;
  (void)member_mask;
  throw sycl::exception(sycl::errc::runtime, "Masked version of select_from_sub_group not "
                        "supported on host device.");
#endif // __SYCL_DEVICE_ONLY__
}

template <typename T>
T shift_sub_group_right(unsigned int member_mask,
                        sycl::sub_group g, T x, unsigned int delta,
                        int logical_sub_group_size = 32) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__SPIR__)
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  T result = __spirv_GroupNonUniformShuffleUp(__spv::Scope::Subgroup, x, delta);
  if ((id - start_index) < delta) {
    result = x;
  }
  return result;
#elif defined(__NVPTX__)
  T result;
  int cVal = ((32 - logical_sub_group_size) << 8);
  SHFL_SYNC(result, member_mask, x, delta, cVal, up_i32)
  return result;
#endif
#else
  (void)g;
  (void)x;
  (void)delta;
  (void)logical_sub_group_size;
  (void)member_mask;
  throw sycl::exception(sycl::errc::runtime, "Masked version of shift_sub_group_right not "
                        "supported on host device.");
#endif // __SYCL_DEVICE_ONLY__
}

/**
 * Warp synchronous shfl_up
 */
inline unsigned int SHFL_UP_SYNC(unsigned int word, int src_offset,
                                          int flags, unsigned int member_mask,
                                          const sycl::nd_item<3> &item)
{
#ifdef PTX
    asm volatile("shfl.sync.up.b32 %0, %1, %2, %3, %4;"
        : "=r"(word) : "r"(word), "r"(src_offset), "r"(flags), "r"(member_mask));
#else
    //word = shift_sub_group_right(member_mask, item.get_sub_group(), word, src_offset, flags);
    word = sycl::shift_group_right(item.get_sub_group(), word, src_offset);
#endif
    return word;
}

template <int LOGICAL_WARP_THREADS, ///< Number of threads per logical warp
          typename T>
inline T ShuffleUp(
    T input,        ///< [in] The value to broadcast
    int src_offset, ///< [in] The relative down-offset of the peer to read from
    int first_thread, ///< [in] Index of first lane in logical warp (typically
                      ///< 0)
    unsigned int member_mask,///< [in] 32-bit mask of participating warp lanes
    const sycl::nd_item<3> &item)
{
    /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    enum {
        SHFL_C = (32 - LOGICAL_WARP_THREADS) << 8
    };

    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    unsigned int shuffle_word;
    shuffle_word = SHFL_UP_SYNC((unsigned int)input_alias[0], src_offset,
                                first_thread | SHFL_C, member_mask, item);
    output_alias[0] = shuffle_word;

    #pragma unroll
    for (int WORD = 1; WORD < WORDS; ++WORD)
    {
        shuffle_word =
            SHFL_UP_SYNC((unsigned int)input_alias[WORD], src_offset,
                         first_thread | SHFL_C, member_mask, item);
        output_alias[WORD] = shuffle_word;
    }

    return output;
}

/**
 * Warp synchronous shfl_idx
 */
inline unsigned int SHFL_IDX_SYNC(unsigned int word, int src_lane,
                                           int flags, unsigned int member_mask,
                                           const sycl::nd_item<3> &item)
{
#ifdef PTX
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, %3, %4;"
        : "=r"(word) : "r"(word), "r"(src_lane), "r"(flags), "r"(member_mask));
#else
    //word = select_from_sub_group(member_mask, item.get_sub_group(), word, src_lane, flags);
    word = sycl::select_from_group(item.get_sub_group(), word, src_lane);
#endif
    return word;
}

template <int LOGICAL_WARP_THREADS, ///< Number of threads per logical warp
          typename T>
inline T
ShuffleIndex(T input,      ///< [in] The value to broadcast
             int src_lane, ///< [in] Which warp lane is to do the broadcasting
             unsigned int member_mask,///< [in] 32-bit mask of participating warp lanes
             const sycl::nd_item<3> &item)
{
    /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
    enum {
        SHFL_C = ((32 - LOGICAL_WARP_THREADS) << 8) | (LOGICAL_WARP_THREADS - 1)
    };

    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    unsigned int shuffle_word;
    shuffle_word = SHFL_IDX_SYNC((unsigned int)input_alias[0], src_lane, SHFL_C,
                                 member_mask, item);

    output_alias[0] = shuffle_word;

    #pragma unroll
    for (int WORD = 1; WORD < WORDS; ++WORD)
    {
        shuffle_word = SHFL_IDX_SYNC((unsigned int)input_alias[WORD], src_lane,
                                     SHFL_C, member_mask, item);

        output_alias[WORD] = shuffle_word;
    }

    return output;
}

/**
 * \brief WarpScanShfl provides SHFL-based variants of parallel prefix scan of items partitioned across a CUDA thread warp.
 *
 * LOGICAL_WARP_THREADS must be a power-of-two
 */
template <
    typename    T,                      ///< Data type being scanned
    int         LOGICAL_WARP_THREADS>   ///< Number of threads per logical warp
struct WarpScanShfl
{
    //---------------------------------------------------------------------
    // Constants and type definitions
    //---------------------------------------------------------------------

    enum
    {
        /// Whether the logical warp size and the PTX warp size coincide
        IS_ARCH_WARP = (LOGICAL_WARP_THREADS == WARP_THREADS(0)),

        /// The number of warp scan steps
        STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

        /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
        SHFL_C = (WARP_THREADS(0) - LOGICAL_WARP_THREADS) << 8
    };

    template <typename S>
    struct IntegerTraits
    {
        enum {
            ///Whether the data type is a small (32b or less) integer for which we can use a single SFHL instruction per exchange
            IS_SMALL_UNSIGNED = (Traits<S>::CATEGORY == UNSIGNED_INTEGER) && (sizeof(S) <= sizeof(unsigned int))
        };
    };

    /// Shared memory storage layout type
    struct TempStorage {};


    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    /// Lane index in logical warp
    unsigned int lane_id;

    /// Logical warp index in 32-thread physical warp
    unsigned int warp_id;

    /// 32-thread physical warp member mask of logical warp
    unsigned int member_mask;

    const sycl::nd_item<3> &item;
    //---------------------------------------------------------------------
    // Construction
    //---------------------------------------------------------------------

    /// Constructor
    explicit inline
    WarpScanShfl(TempStorage &,
                 const sycl::nd_item<3> &item /*temp_storage*/)
        : lane_id(LaneId(item)),
          warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS)),
          member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id)),
          item(item)
    {
        if (!IS_ARCH_WARP)
        {
            lane_id = lane_id % LOGICAL_WARP_THREADS;
        }
    }


    //---------------------------------------------------------------------
    // Inclusive scan steps
    //---------------------------------------------------------------------

    /// Inclusive prefix scan step (specialized for summation across int32 types)
    inline int InclusiveScanStep(
        int input,       ///< [in] Calling thread's input item.
        Sum /*scan_op*/, ///< [in] Binary scan operator
        int first_lane,  ///< [in] Index of first lane in segment
        int offset)      ///< [in] Up-offset to pull from
    {
        int output = 0;
        //int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

        output = input;
        //int value = shift_sub_group_right(member_mask, item.get_sub_group(), output, offset, shfl_c);
        int value = sycl::shift_group_right(item.get_sub_group(), output, offset);
        if (lane_id >= offset) output += value;

        return output;
    }

    /// Inclusive prefix scan step (specialized for summation across uint32 types)
    inline unsigned int InclusiveScanStep(
        unsigned int input, ///< [in] Calling thread's input item.
        Sum /*scan_op*/,    ///< [in] Binary scan operator
        int first_lane,     ///< [in] Index of first lane in segment
        int offset)         ///< [in] Up-offset to pull from
    {
        unsigned int output = 0;

#ifdef PTX
        int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)
        // Use predicate set from SHFL to guard against invalid peers
        asm volatile(
            "{"
            "  .reg .u32 r0;"
            "  .reg .pred p;"
            "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
            "  @p add.u32 r0, r0, %4;"
            "  mov.u32 %0, r0;"
            "}"
            : "=r"(output) : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));

#else
        output = input;
        //unsigned int value = shift_sub_group_right(member_mask, item.get_sub_group(), output, offset, shfl_c);
        unsigned int value = sycl::shift_group_right(item.get_sub_group(), output, offset);
        if (lane_id >= offset) output += value;
#endif
        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across fp32 types)
    inline float InclusiveScanStep(
        float input,     ///< [in] Calling thread's input item.
        Sum /*scan_op*/, ///< [in] Binary scan operator
        int first_lane,  ///< [in] Index of first lane in segment
        int offset)      ///< [in] Up-offset to pull from
    {
        float output = 0;

#ifdef PTX
        int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)
        // Use predicate set from SHFL to guard against invalid peers
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
            "  @p add.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(shfl_c), "f"(input), "r"(member_mask));
#else
        output = input;
        //float value = shift_sub_group_right(member_mask, item.get_sub_group(), output, offset, shfl_c);
        float value = sycl::shift_group_right(item.get_sub_group(), output, offset);
        if (lane_id >= offset) output += value;
#endif

        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across unsigned long long types)
    inline unsigned long long InclusiveScanStep(
        unsigned long long input, ///< [in] Calling thread's input item.
        Sum /*scan_op*/,          ///< [in] Binary scan operator
        int first_lane,           ///< [in] Index of first lane in segment
        int offset)               ///< [in] Up-offset to pull from
    {
        unsigned long long output = 0;

#ifdef PTX
        int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)
        // Use predicate set from SHFL to guard against invalid peers
        asm volatile(
            "{"
            "  .reg .u64 r0;"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
            "  shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.u64 r0, r0, %4;"
            "  mov.u64 %0, r0;"
            "}"
            : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "l"(input), "r"(member_mask));
#else
        output = input;
        //unsigned long long value = shift_sub_group_right(member_mask, item.get_sub_group(), output, offset, shfl_c);
        unsigned long long value = sycl::shift_group_right(item.get_sub_group(), output, offset);
        if (lane_id >= offset) output += value;
#endif
        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across long long types)
    inline long long InclusiveScanStep(
        long long input, ///< [in] Calling thread's input item.
        Sum /*scan_op*/, ///< [in] Binary scan operator
        int first_lane,  ///< [in] Index of first lane in segment
        int offset)      ///< [in] Up-offset to pull from
    {
        long long output = 0;

#ifdef PTX
        int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)
        // Use predicate set from SHFL to guard against invalid peers
        asm volatile(
            "{"
            "  .reg .s64 r0;"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
            "  shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.s64 r0, r0, %4;"
            "  mov.s64 %0, r0;"
            "}"
            : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "l"(input), "r"(member_mask));
#else
        output = input;
        //long long value = shift_sub_group_right(member_mask, item.get_sub_group(), output, offset, shfl_c);
        long long value = sycl::shift_group_right(item.get_sub_group(), output, offset);
        if (lane_id >= offset) output += value;
#endif

        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across fp64 types)
    inline double InclusiveScanStep(
        double input,    ///< [in] Calling thread's input item.
        Sum /*scan_op*/, ///< [in] Binary scan operator
        int first_lane,  ///< [in] Index of first lane in segment
        int offset)      ///< [in] Up-offset to pull from
    {
        double output = 0;

#ifdef PTX
        int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)
        // Use predicate set from SHFL to guard against invalid peers
        asm volatile(
            "{"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  .reg .f64 r0;"
            "  mov.b64 %0, %1;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.sync.up.b32 lo|p, lo, %2, %3, %4;"
            "  shfl.sync.up.b32 hi|p, hi, %2, %3, %4;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.f64 %0, %0, r0;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));
#else
        output = input;
        //double value = shift_sub_group_right(member_mask, item.get_sub_group(), output, offset, shfl_c);
        double value = sycl::shift_group_right(item.get_sub_group(), output, offset);
        if (lane_id >= offset) output += value;
#endif
        return output;
    }


/*
    /// Inclusive prefix scan (specialized for ReduceBySegmentOp<Sum> across KeyValuePair<OffsetT, Value> types)
    template <typename Value, typename OffsetT>
    __device__ __forceinline__ KeyValuePair<OffsetT, Value>InclusiveScanStep(
        KeyValuePair<OffsetT, Value>    input,              ///< [in] Calling thread's input item.
        ReduceBySegmentOp<Sum>     scan_op,            ///< [in] Binary scan operator
        int                             first_lane,         ///< [in] Index of first lane in segment
        int                             offset)             ///< [in] Up-offset to pull from
    {
        KeyValuePair<OffsetT, Value> output;

        output.value = InclusiveScanStep(input.value, Sum(), first_lane, offset, Int2Type<IntegerTraits<Value>::IS_SMALL_UNSIGNED>());
        output.key = InclusiveScanStep(input.key, Sum(), first_lane, offset, Int2Type<IntegerTraits<OffsetT>::IS_SMALL_UNSIGNED>());

        if (input.key > 0)
            output.value = input.value;

        return output;
    }
*/

    /// Inclusive prefix scan step (generic)
    template <typename _T, typename ScanOpT>
    inline _T InclusiveScanStep(
        _T input,        ///< [in] Calling thread's input item.
        ScanOpT scan_op, ///< [in] Binary scan operator
        int first_lane,  ///< [in] Index of first lane in segment
        int offset)      ///< [in] Up-offset to pull from
    {
        _T temp = ShuffleUp<LOGICAL_WARP_THREADS>(input, offset, first_lane, member_mask, item);

        // Perform scan op if from a valid peer
        _T output = scan_op(temp, input);
        if (static_cast<int>(lane_id) < first_lane + offset)
            output = input;

        return output;
    }


    /// Inclusive prefix scan step (specialized for small integers size 32b or less)
    template <typename _T, typename ScanOpT>
    inline _T InclusiveScanStep(
        _T input,        ///< [in] Calling thread's input item.
        ScanOpT scan_op, ///< [in] Binary scan operator
        int first_lane,  ///< [in] Index of first lane in segment
        int offset,      ///< [in] Up-offset to pull from
        Int2Type<true>
/*is_small_unsigned*/) ///< [in] Marker type indicating
                                             ///< whether T is a small integer
    {
        return InclusiveScanStep(input, scan_op, first_lane, offset);
    }


    /// Inclusive prefix scan step (specialized for types other than small integers size 32b or less)
    template <typename _T, typename ScanOpT>
    inline _T InclusiveScanStep(
        _T input,        ///< [in] Calling thread's input item.
        ScanOpT scan_op, ///< [in] Binary scan operator
        int first_lane,  ///< [in] Index of first lane in segment
        int offset,      ///< [in] Up-offset to pull from
        Int2Type<false>
            /*is_small_unsigned*/) ///< [in] Marker type indicating
                                             ///< whether T is a small integer
    {
        return InclusiveScanStep(input, scan_op, first_lane, offset);
    }


    /******************************************************************************
     * Interface
     ******************************************************************************/

    //---------------------------------------------------------------------
    // Broadcast
    //---------------------------------------------------------------------

    /// Broadcast
    inline T
    Broadcast(T input, ///< [in] The value to broadcast
              int src_lane) ///< [in] Which warp lane is to do the broadcasting
    {
        return ShuffleIndex<LOGICAL_WARP_THREADS>(input, src_lane, member_mask,
                                                  item);
    }


    //---------------------------------------------------------------------
    // Inclusive operations
    //---------------------------------------------------------------------

    /// Inclusive scan
    template <typename _T, typename ScanOpT>
    inline void InclusiveScan(
        _T input,             ///< [in] Calling thread's input item.
        _T &inclusive_output, ///< [out] Calling thread's output item.  May be
                              ///< aliased with \p input.
        ScanOpT scan_op)      ///< [in] Binary scan operator
    {
        inclusive_output = input;

        // Iterate scan steps
        int segment_first_lane = 0;

        // Iterate scan steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            inclusive_output = InclusiveScanStep(
                inclusive_output, scan_op, segment_first_lane, (1 << STEP),
                Int2Type<IntegerTraits<T>::IS_SMALL_UNSIGNED>());
        }

    }

/*
    /// Inclusive scan, specialized for reduce-value-by-key
    template <typename KeyT, typename ValueT, typename ReductionOpT>
    __device__ __forceinline__ void InclusiveScan(
        KeyValuePair<KeyT, ValueT>      input,              ///< [in] Calling thread's input item.
        KeyValuePair<KeyT, ValueT>      &inclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
        ReduceByKeyOp<ReductionOpT >    scan_op)            ///< [in] Binary scan operator
    {
        inclusive_output = input;

        KeyT pred_key = ShuffleUp<LOGICAL_WARP_THREADS>(inclusive_output.key, 1, 0, member_mask);

        unsigned int ballot = WARP_BALLOT((pred_key != inclusive_output.key), member_mask);

        // Mask away all lanes greater than ours
        ballot = ballot & LaneMaskLe();

        // Find index of first set bit
        int segment_first_lane = MAX(0, 31 - __clz(ballot));

        // Iterate scan steps
        #pragma unroll
        for (int STEP = 0; STEP < STEPS; STEP++)
        {
            inclusive_output.value = InclusiveScanStep(
                inclusive_output.value,
                scan_op.op,
                segment_first_lane,
                (1 << STEP),
                Int2Type<IntegerTraits<T>::IS_SMALL_UNSIGNED>());
        }
    }
*/


    /// Inclusive scan with aggregate
    template <typename ScanOpT>
    inline void InclusiveScan(
        T input,             ///< [in] Calling thread's input item.
        T &inclusive_output, ///< [out] Calling thread's output item.  May be
                             ///< aliased with \p input.
        ScanOpT scan_op,     ///< [in] Binary scan operator
        T &warp_aggregate)   ///< [out] Warp-wide aggregate reduction of input items.
    {
        InclusiveScan(input, inclusive_output, scan_op);

        // Grab aggregate from last warp lane
        warp_aggregate = ShuffleIndex<LOGICAL_WARP_THREADS>(
            inclusive_output, LOGICAL_WARP_THREADS - 1, member_mask, item);
    }


    //---------------------------------------------------------------------
    // Get exclusive from inclusive
    //---------------------------------------------------------------------

    /// Update inclusive and exclusive using input and inclusive
    template <typename ScanOpT, typename IsIntegerT>
    inline void
    Update(T /*input*/,         ///< [in]
           T &inclusive,        ///< [in, out]
           T &exclusive,        ///< [out]
           ScanOpT /*scan_op*/, ///< [in]
           IsIntegerT /*is_integer*/) ///< [in]
    {
        // initial value unknown
        exclusive = ShuffleUp<LOGICAL_WARP_THREADS>(inclusive, 1, 0, member_mask, item);
    }

    /// Update inclusive and exclusive using input and inclusive (specialized for summation of integer types)
    inline void Update(T input, T &inclusive, T &exclusive,
                                Sum /*scan_op*/, Int2Type<true>) /*is_integer*/
    {
        // initial value presumed 0
        exclusive = inclusive - input;
    }

    /// Update inclusive and exclusive using initial value using input, inclusive, and initial value
    template <typename ScanOpT, typename IsIntegerT>
    inline void Update(T /*input*/, T &inclusive, T &exclusive,
                                ScanOpT scan_op, T initial_value, IsIntegerT)
    {
        inclusive = scan_op(initial_value, inclusive);
        exclusive = ShuffleUp<LOGICAL_WARP_THREADS>(inclusive, 1, 0, member_mask, item);

        if (lane_id == 0)
            exclusive = initial_value;
    }

    /// Update inclusive and exclusive using initial value using input and inclusive (specialized for summation of integer types)
    inline void Update(T input, T &inclusive, T &exclusive,
                                Sum scan_op, T initial_value,
                                Int2Type<true> /*is_integer*/)
    {
        inclusive = scan_op(initial_value, inclusive);
        exclusive = inclusive - input;
    }


    /// Update inclusive, exclusive, and warp aggregate using input and inclusive
    template <typename ScanOpT, typename IsIntegerT>
    inline void Update(T input, T &inclusive, T &exclusive,
                                T &warp_aggregate, ScanOpT scan_op,
                                IsIntegerT is_integer)
    {
        warp_aggregate = ShuffleIndex<LOGICAL_WARP_THREADS>(
            inclusive, LOGICAL_WARP_THREADS - 1, member_mask, item);
        Update(input, inclusive, exclusive, scan_op, is_integer);
    }

    /// Update inclusive, exclusive, and warp aggregate using input, inclusive, and initial value
    template <typename ScanOpT, typename IsIntegerT>
    inline void Update(T input, T &inclusive, T &exclusive,
                                T &warp_aggregate, ScanOpT scan_op,
                                T initial_value, IsIntegerT is_integer)
    {
        warp_aggregate = ShuffleIndex<LOGICAL_WARP_THREADS>(
            inclusive, LOGICAL_WARP_THREADS - 1, member_mask, item);
        Update(input, inclusive, exclusive, scan_op, initial_value, is_integer);
    }
};
