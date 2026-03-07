#include <chrono>
#include <future>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h> // For thrust::device
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <hip/hip_runtime.h>
#include <cstdlib>


// This example shows how to execute a Thrust device algorithm on an explicit
// HIP stream. The simple program below fills a vector with the numbers
// [0, 1000) (thrust::sequence) and then performs a scan operation
// (thrust::inclusive_scan) on them. Both algorithms are executed on the same
// custom HIP stream using the HIP execution policies.
//
// Thrust provides two execution policies that accept HIP streams that differ
// in when/if they synchronize the stream:
// 1. thrust::hip::par.on(stream)
//      - `stream` will *always* be synchronized before an algorithm returns.
//      - This is the default `thrust::device` policy when compiling with the
//        HIP device backend.
// 2. thrust::hip::par_nosync.on(stream)
//      - `stream` will only be synchronized when necessary for correctness
//        (e.g., returning a result from `thrust::reduce`). This is a hint that
//        may be ignored by an algorithm's implementation.

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  std::cout << "Thrust version: " << THRUST_VERSION << "\n";

  // Create the stream:
  hipStream_t s;
  hipError_t err = hipStreamCreate(&s);
  if (err != hipSuccess)
  {
    std::cerr << "Error creating stream: " << hipGetErrorString(err) << "\n";
    return 1;
  }

  int sum = -1;

  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < repeat; i++)
  {
    thrust::device_vector<int> d_vec(n);
    thrust::device_vector<int> d_res(n);

    // Construct a new `nosync` execution policy with the custom stream
    auto nosync_exec_policy = thrust::hip::par_nosync.on(s);

    // Fill the vector with sequential data.
    // This will execute using the custom stream and the stream will *not* be
    // synchronized before the function returns, meaning asynchronous work may
    // still be executing after returning and the contents of `d_vec` are
    // undefined. Synchronization is not needed here because the following
    // `inclusive_scan` is executed on the same stream and is therefore guaranteed
    // to be ordered after the `sequence`
    thrust::sequence(nosync_exec_policy, d_vec.begin(), d_vec.end());

    // Construct a new *synchronous* execution policy with the same custom stream
    auto sync_exec_policy = thrust::hip::par.on(s);

    auto begin = d_vec.begin();
    auto end = d_vec.end();
    auto binary_op = thrust::plus<int>();
    // std::async captures the algorithm parameters by value
    // use std::launch::async to ensure the creation of a new thread
    std::future<int> future_result = std::async(std::launch::async, [=]
    {
      return thrust::reduce(begin, end, 0, binary_op);
    });

    // Compute in-place inclusive sum scan of data in the vector.
    // This also executes in the custom stream, but the execution policy ensures
    // the stream is synchronized before the algorithm returns. This guarantees
    // there is no pending asynchronous work and the contents of `d_vec` are
    // immediately accessible.
    thrust::inclusive_scan(sync_exec_policy,
                           d_vec.cbegin(),
                           d_vec.cend(),
                           d_res.begin());

    // wait on the result and check that it is correct
    sum = future_result.get() - 
           // This access is only valid because the stream has been synchronized
          d_res.back();

  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time: " << (time * 1e-3f) / repeat << " (us)\n";

  // Free the stream:
  err = hipStreamDestroy(s);
  if (err != hipSuccess)
  {
    std::cerr << "Error destroying stream: " << hipGetErrorString(err) << "\n";
    return 1;
  }

  // Print the sum:
  std::cout << ((sum == 0) ? "PASS" : "FAIL") << "\n";
  if (sum != 0) exit(1);

  return 0;
}
