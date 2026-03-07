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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Test of DeviceRunLengthEncode
 ******************************************************************************/

// Ensure printing of HIP runtime errors to console
#define HIPCUB_STDERR

#include <chrono>
#include <cstdio>
#include <random>
#include <hipcub/hipcub.hpp>
#include "test_util.h"
#include <cstdlib>

using namespace hipcub;

// not supported by hipCUB
#define CubDebugExit(s) s

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool g_verbose          = false;
int g_timing_iterations = 0;
CachingDeviceAllocator g_allocator(true);

/**
 * Dispatch to run-length encode entrypoint
 */
template <typename InputIteratorT,
          typename UniqueOutputIteratorT,
          typename LengthsOutputIteratorT,
          typename NumRunsIterator,
          typename OffsetT>
__forceinline__ hipError_t Dispatch(int timing_iterations,
                                    void *d_temp_storage,
                                    size_t &temp_storage_bytes,
                                    InputIteratorT d_in,
                                    UniqueOutputIteratorT d_unique_out,
                                    LengthsOutputIteratorT d_lengths_out,
                                    NumRunsIterator d_num_runs,
                                    OffsetT num_items)
{
  hipError_t error = hipSuccess;
  for (int i = 0; i < timing_iterations; ++i)
  {
    error = DeviceRunLengthEncode::Encode(d_temp_storage,
                                          temp_storage_bytes,
                                          d_in,
                                          d_unique_out,
                                          d_lengths_out,
                                          d_num_runs,
                                          num_items);
    hipDeviceSynchronize();
  }
  return error;
}


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Initialize problem
 */
template <typename T>
void Initialize(int entropy_reduction, T *h_in, int num_items, int max_segment)
{
  unsigned int max_int = (unsigned int)-1;

  int key = 0;
  int i   = 0;
  while (i < num_items)
  {
    // Select number of repeating occurrences for the current run
    int repeat;
    if (max_segment < 0)
    {
      repeat = num_items;
    }
    else if (max_segment < 2)
    {
      repeat = 1;
    }
    else
    {
      std::mt19937 mt(19937+i);
      std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
      repeat = dist(mt);
      for (int n = 1; n <= entropy_reduction; n++) {
        repeat &= dist(mt);
      }
      repeat = (int)((double(repeat) * double(max_segment)) / double(max_int));
      repeat = CUB_MAX(1, repeat);
    }

    int j = i;
    while (j < CUB_MIN(i + repeat, num_items))
    {
      h_in[j] = key;
      j++;
    }

    i = j;
    key++;
  }

  if (g_verbose)
  {
    printf("Input:\n");
    DisplayResults(h_in, num_items);
    printf("\n\n");
  }
}

/**
 * Solve problem.  Returns total number of segments identified
 */
template <typename InputIteratorT,
          typename T,
          typename OffsetT,
          typename LengthT>
int Solve(InputIteratorT h_in,
          T *h_unique_reference,
          OffsetT *h_offsets_reference,
          LengthT *h_lengths_reference,
          int num_items)
{
  if (num_items == 0)
    return 0;

  // First item
  T previous     = h_in[0];
  LengthT length = 1;
  int num_runs   = 0;
  int run_begin  = 0;

  // Subsequent items
  for (int i = 1; i < num_items; ++i)
  {
    if (previous != h_in[i])
    {
        h_unique_reference[num_runs]  = previous;
        h_offsets_reference[num_runs] = run_begin;
        h_lengths_reference[num_runs] = length;
        num_runs++;
      length    = 1;
      run_begin = i;
    }
    else
    {
      length++;
    }
    previous = h_in[i];
  }

  {
    h_unique_reference[num_runs]  = previous;
    h_offsets_reference[num_runs] = run_begin;
    h_lengths_reference[num_runs] = length;
    num_runs++;
  }

  return num_runs;
}

/**
 * Test DeviceRunLengthEncode for a given problem input
 */
template <typename DeviceInputIteratorT,
          typename T,
          typename OffsetT,
          typename LengthT>
void Test(DeviceInputIteratorT d_in,
          T *h_unique_reference,
          OffsetT *h_offsets_reference,
          LengthT *h_lengths_reference,
          int num_runs,
          int num_items)
{
  // Allocate device output arrays and number of segments
  T *d_unique_out        = NULL;
  OffsetT *d_lengths_out = NULL;
  int *d_num_runs        = NULL;

  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_unique_out, sizeof(T) * num_items));
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_lengths_out, sizeof(LengthT) * num_items));
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_num_runs, sizeof(int)));

  // Allocate temporary storage
  void *d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(Dispatch(1,
                        d_temp_storage,
                        temp_storage_bytes,
                        d_in,
                        d_unique_out,
                        d_lengths_out,
                        d_num_runs,
                        num_items));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Clear device output arrays
  CubDebugExit(hipMemset(d_unique_out, 0, sizeof(T) * num_items));
  CubDebugExit(hipMemset(d_lengths_out, 0, sizeof(LengthT) * num_items));
  CubDebugExit(hipMemset(d_num_runs, 0, sizeof(int)));

  // Run warmup/correctness iteration
  CubDebugExit(Dispatch(1,
                        d_temp_storage,
                        temp_storage_bytes,
                        d_in,
                        d_unique_out,
                        d_lengths_out,
                        d_num_runs,
                        num_items));

  // Check for correctness (and display results, if specified)
  int compare0 = 0;
  int compare1 = 0;
  int compare2 = 0;

  compare0 = CompareDeviceResults(h_unique_reference, d_unique_out, num_runs, true, g_verbose);
  printf("\t Keys %s\n", compare0 ? "FAIL" : "PASS");

  compare1 = CompareDeviceResults(h_lengths_reference, d_lengths_out, num_runs, true, g_verbose);
  printf("\t Lengths %s\n", compare1 ? "FAIL" : "PASS");

  compare2 = CompareDeviceResults(&num_runs, d_num_runs, 1, true, g_verbose);
  printf("\t Count %s\n", compare2 ? "FAIL" : "PASS");

  // Flush any stdout/stderr
  fflush(stdout);
  fflush(stderr);

  // Performance
  auto start = std::chrono::steady_clock::now();

  CubDebugExit(Dispatch(g_timing_iterations,
                        d_temp_storage,
                        temp_storage_bytes,
                        d_in,
                        d_unique_out,
                        d_lengths_out,
                        d_num_runs,
                        num_items));

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  float elapsed_millis = time * 1e-6f;

  // Display performance
  if (g_timing_iterations > 0)
  {
    float avg_millis = elapsed_millis / g_timing_iterations;
    float giga_rate  = float(num_items) / avg_millis / 1000.0f / 1000.0f;
    int bytes_moved  = (num_items * sizeof(T)) + (num_runs * (sizeof(OffsetT) + sizeof(LengthT)));
    float giga_bandwidth = float(bytes_moved) / avg_millis / 1000.0f / 1000.0f;
    printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s",
           avg_millis,
           giga_rate,
           giga_bandwidth);
  }
  printf("\n\n");

  // Flush any stdout/stderr
  fflush(stdout);
  fflush(stderr);

  // Cleanup
  if (d_unique_out)
    CubDebugExit(g_allocator.DeviceFree(d_unique_out));
  if (d_lengths_out)
    CubDebugExit(g_allocator.DeviceFree(d_lengths_out));
  if (d_num_runs)
    CubDebugExit(g_allocator.DeviceFree(d_num_runs));
  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

/**
 * Test DeviceRunLengthEncode on pointer type
 */
template <typename T, typename OffsetT, typename LengthT>
void TestPointer(int num_items, int entropy_reduction, int max_segment)
{
  // Allocate host arrays
  T *h_in                      = new T[num_items];
  T *h_unique_reference        = new T[num_items];
  OffsetT *h_offsets_reference = new OffsetT[num_items];
  LengthT *h_lengths_reference = new LengthT[num_items];

  for (int i = 0; i < num_items; ++i)
    h_offsets_reference[i] = 1;

  // Initialize problem and solution
  Initialize(entropy_reduction, h_in, num_items, max_segment);

  int num_runs = Solve(h_in,
                       h_unique_reference,
                       h_offsets_reference,
                       h_lengths_reference,
                       num_items);

  printf("\nTest pointer: %d items, %d segments (avg run length %.3f), {%s key, %s offset, "
         "%s length}, max_segment %d, entropy_reduction %d\n",
         num_items,
         num_runs,
         float(num_items) / num_runs,
         typeid(T).name(),
         typeid(OffsetT).name(),
         typeid(LengthT).name(),
         max_segment,
         entropy_reduction);
  fflush(stdout);

  // Allocate problem device arrays
  T *d_in = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_in, sizeof(T) * num_items));

  // Initialize device input
  CubDebugExit(hipMemcpy(d_in, h_in, sizeof(T) * num_items, hipMemcpyHostToDevice));

  // Run Test
  Test(d_in,
       h_unique_reference,
       h_offsets_reference,
       h_lengths_reference,
       num_runs,
       num_items);

  // Cleanup
  if (h_in)
    delete[] h_in;
  if (h_unique_reference)
    delete[] h_unique_reference;
  if (h_offsets_reference)
    delete[] h_offsets_reference;
  if (h_lengths_reference)
    delete[] h_lengths_reference;
  if (d_in)
    CubDebugExit(g_allocator.DeviceFree(d_in));
}

/**
 * Test on iterator type
 */
template <typename T, typename OffsetT, typename LengthT>
void TestIterator(int num_items)
{
  // Allocate host arrays
  T *h_unique_reference        = new T[num_items];
  OffsetT *h_offsets_reference = new OffsetT[num_items];
  LengthT *h_lengths_reference = new LengthT[num_items];

  T one_val = 1;
  ConstantInputIterator<T, int> h_in(one_val);

  // Initialize problem and solution
  int num_runs = Solve(h_in,
                       h_unique_reference,
                       h_offsets_reference,
                       h_lengths_reference,
                       num_items);

  printf("\nTest iterator: on %d items, %d segments (avg run length %.3f), {%s key, %s "
         "offset, %s length}\n",
         num_items,
         num_runs,
         float(num_items) / num_runs,
         typeid(T).name(),
         typeid(OffsetT).name(),
         typeid(LengthT).name());
  fflush(stdout);

  // Run Test
  Test(h_in,
       h_unique_reference,
       h_offsets_reference,
       h_lengths_reference,
       num_runs,
       num_items);

  // Cleanup
  if (h_unique_reference)
    delete[] h_unique_reference;
  if (h_offsets_reference)
    delete[] h_offsets_reference;
  if (h_lengths_reference)
    delete[] h_lengths_reference;
}

/**
 * Test different gen modes
 */
template <typename T, typename OffsetT, typename LengthT>
void Test(int num_items)
{
  // Test iterator (one run)
  TestIterator<T, OffsetT, LengthT>(num_items);

  // Evaluate different run lengths / segment sizes
  const int max_seg_limit = CUB_MIN(num_items, 1 << 16);
  const int max_seg_inc   = 4;
  for (int max_segment = 1, entropy_reduction = 0; max_segment <= max_seg_limit;
       max_segment <<= max_seg_inc, entropy_reduction++)
  {
    const int max_seg = CUB_MAX(1, max_segment);
    TestPointer<T, OffsetT, LengthT>(num_items, entropy_reduction, max_seg);
  }
}

/**
 * Test different dispatch
 */
template <typename T, typename OffsetT, typename LengthT>
void TestDispatch(int num_items)
{
  Test<T, OffsetT, LengthT>(num_items);
}

/**
 * Test different input sizes
 */
template <typename T, typename OffsetT, typename LengthT>
void TestSize(int num_items)
{
  if (num_items < 0)
  {
    //TestDispatch<T, OffsetT, LengthT>(0);
    //TestDispatch<T, OffsetT, LengthT>(1);
    //TestDispatch<T, OffsetT, LengthT>(100);
    //TestDispatch<T, OffsetT, LengthT>(10000);
    TestDispatch<T, OffsetT, LengthT>(1000000);
  }
  else
  {
    TestDispatch<T, OffsetT, LengthT>(num_items);
  }
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

int main(int argc, char **argv)
{
  int num_items = -1;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");
  args.GetCmdLineArgument("n", num_items);
  args.GetCmdLineArgument("i", g_timing_iterations);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--n=<input items> "
           "[--i=<timing iterations> "
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Initialize device
  printf("\n");

  // Test different input types
  // TestSize<signed char, int, int>(num_items);
  // TestSize<short, int, int>(num_items);
  TestSize<int, int, int>(num_items);
  // TestSize<long, int, int>(num_items);
  // TestSize<long long, int, int>(num_items);
  // TestSize<float, int, int>(num_items);
  // TestSize<double, int, int>(num_items);

  if (compare0) return 1;
  if (compare1) return 1;
  if (compare2) return 1;
  return 0;
}
