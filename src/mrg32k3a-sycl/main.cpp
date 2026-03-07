/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <chrono>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"
#include <cstdlib>

using namespace oneapi::mkl::rng;

void print_vector(const std::vector<float> &data, size_t n) {
  for (size_t i = data.size()-n; i < data.size(); i++)
    std::printf("%0.6f\n", data[i]);
}

void print_vector(const std::vector<unsigned int> &data, size_t n) {
  for (size_t i = data.size()-n; i < data.size(); i++)
    std::printf("%d\n", data[i]);
}

using data_type = float;

void run_on_device(const int &n, const unsigned long long &offset,
                   const unsigned long long &seed,
                   const mrg32k3a_mode::optimal &order,
                   sycl::queue &q,
                   std::vector<data_type> &h_data) {

  /* C data to device */
  data_type *d_data = sycl::malloc_device<data_type>(h_data.size(), q);

  mrg32k3a engine (q, seed, order); 
  
  /* Set offset */
  skip_ahead(engine, offset);

  uniform<data_type> distribution;

  /* Generate n floats on host */
  try {
    generate(distribution, engine, n, d_data).wait();
  }
  catch (sycl::exception const& e) {
      std::cout << "\t\tSYCL exception during oneapi::mkl::rng::generate() call\n"
                << e.what() << std::endl;
      return;
  }
  catch (oneapi::mkl::exception const& e) {
      std::cout << "\toneMKL exception during oneapi::mkl::rng::generate() call\n"
                << e.what() << std::endl;
      return;
  }

  /* Copy data to host */
  q.memcpy(h_data.data(), d_data,
                        sizeof(data_type) * h_data.size()).wait();

  /* Cleanup */
  sycl::free(d_data, q);
}

void run_on_host(const int &n, const unsigned long long &offset,
                 const unsigned long long &seed,
                 const mrg32k3a_mode::optimal &order,
                 sycl::queue &q,
                 std::vector<data_type> &h_data) {

  mrg32k3a engine (q, seed, order); 
  
  /* Set offset */
  skip_ahead(engine, offset);

  uniform<data_type> distribution;

  /* Generate n floats on host */
  try {
    generate(distribution, engine, n, h_data.data()).wait();
  }
  catch (sycl::exception const& e) {
      std::cout << "\t\tSYCL exception during oneapi::mkl::rng::generate() call\n"
                << e.what() << std::endl;
      return;
  }
  catch (oneapi::mkl::exception const& e) {
      std::cout << "\toneMKL exception during oneapi::mkl::rng::generate() call\n"
                << e.what() << std::endl;
      return;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of pseudorandom numbers to generate> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  mrg32k3a_mode::optimal order = mrg32k3a_mode::optimal_v;

  const unsigned long long offset = 0ULL;
  const unsigned long long seed = 1234ULL;

  /* Allocate n floats on host */
  std::vector<data_type> h_data(n, 0);
  std::vector<data_type> d_data(n, -1);

  sycl::queue h_q(sycl::cpu_selector_v, sycl::property::queue::in_order());
  sycl::queue d_q(sycl::gpu_selector_v, sycl::property::queue::in_order());

  // warmup
  run_on_host(n, offset, seed, order, h_q, h_data);
  run_on_device(n, offset, seed, order, d_q, d_data);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    run_on_host(n, offset, seed, order, h_q, h_data);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time on host: %f (us)\n", (time * 1e-3f) / repeat);

#ifdef DEBUG
  printf("Host\n");
  print_vector(h_data, 10);
  printf("=====\n");
#endif

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    run_on_device(n, offset, seed, order, d_q, d_data);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time on device: %f (us)\n", (time * 1e-3f) / repeat);

#ifdef DEBUG
  printf("Device\n");
  print_vector(d_data, 10);
  printf("=====\n");
#endif

  bool ok = true;
  for (int i = 0; i < n; i++) {
    if (std::abs(h_data[i] - d_data[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  return EXIT_SUCCESS;
}
