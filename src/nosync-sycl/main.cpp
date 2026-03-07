#include <oneapi/dpl/numeric>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>  // std::copy(policy, ...)
#include <oneapi/dpl/iterator>   // dpl::begin()
#ifdef ASYNC_API
#include <oneapi/dpl/async>
#endif
#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <cstdlib>

int main(int argc, char *argv[]) try {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  std::cout << "oneDPL version: "
            << ONEDPL_VERSION_MAJOR << "."
            << ONEDPL_VERSION_MINOR << "."
            << ONEDPL_VERSION_PATCH << "\n";

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int sum = -1;

  auto policy = oneapi::dpl::execution::make_device_policy(q);

  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < repeat; i++)
  {
    sycl::buffer<int, 1> d_vec(n);
    sycl::buffer<int, 1> d_res(n);

    auto res_begin = oneapi::dpl::begin(d_res, sycl::write_only, sycl::no_init);
    auto vals_begin = oneapi::dpl::begin(d_vec, sycl::write_only, sycl::no_init);
    auto scan_begin = oneapi::dpl::begin(d_vec, sycl::read_only);
    auto scan_end = oneapi::dpl::end(d_vec, sycl::read_only);
    auto counting_begin = oneapi::dpl::counting_iterator<int>{0};

    #ifdef ASYNC_API
    // Fill the buffer with sequential data.
    auto c = oneapi::dpl::experimental::copy_async(
             policy, counting_begin, counting_begin + n, vals_begin);

    auto r = oneapi::dpl::experimental::reduce_async(
             policy, scan_begin, scan_end, c);

    // Compute inclusive sum scan of data in the vector.
    auto s = oneapi::dpl::experimental::inclusive_scan_async(policy, scan_begin, scan_end, res_begin, c);

    q.submit([&] (sycl::handler &cgh) {
      cgh.depends_on(s);
      auto acc = d_res.get_access<sycl::access::mode::read>(cgh, sycl::range<1>(1), sycl::id<1>(n-1));
      cgh.copy(acc, &sum);
    }).wait();

    sum = r.get() - sum; 

    #else

    std::copy(policy, counting_begin, counting_begin + n, vals_begin);
    oneapi::dpl::inclusive_scan(policy, scan_begin, scan_end, res_begin);
    auto s = oneapi::dpl::reduce(policy, scan_begin, scan_end);
    q.submit([&] (sycl::handler &cgh) {
      auto acc = d_res.get_access<sycl::access::mode::read>(cgh, sycl::range<1>(1), sycl::id<1>(n-1));
      cgh.copy(acc, &sum);
    }).wait();

    sum = s - sum; 

    #endif
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time: " << (time * 1e-3f) / repeat << " (us)\n";

  // Print the sum:
  std::cout << ((sum == 0) ? "PASS" : "FAIL") << "\n";
  if (sum != 0) exit(1);

  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
