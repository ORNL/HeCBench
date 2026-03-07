#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include "reference.h"

std::vector<char> mtf(sycl::queue &q, std::vector<char> &word)
{
  auto policy = oneapi::dpl::execution::make_device_policy(q);

  std::vector<char> h_list(256);
  std::vector<char> h_word(word.size());

  {
    sycl::buffer<char, 1> d_list(256);

    // copy word from host to device
    sycl::buffer<char, 1> d_word (word.data(), word.size());

    // store the mtf result since input word is read-only
    d_word.set_final_data(h_word.data());

    size_t counter;

    auto d_list_beg = oneapi::dpl::begin(d_list);
    auto d_list_end = oneapi::dpl::end(d_list);
    auto d_word_beg = oneapi::dpl::begin(d_word);
    auto d_word_end = oneapi::dpl::end(d_word);

    for (counter = 0; counter < word.size(); counter++)
    {
      // copy list from host to device
      std::copy(policy, h_list.begin(), h_list.end(), d_list_beg);

      // find the location of the symbol in the list
      auto w = word[counter];
      auto iter = oneapi::dpl::find(policy, d_list_beg, d_list_end, w);

      // update the list when the first symbols are not the same
      if (h_list[0] != w)
      {
        // shift the sublist [begin, iter) right by one
        std::copy(policy, d_list_beg, iter, h_list.begin() + 1);
        h_list[0] = w;
      }
    }

    for (counter = 0; counter < h_list.size(); counter++)
    {
      auto iter = oneapi::dpl::find(policy, d_word_beg, d_word_end, h_list[counter]);
      while (iter != d_word_end)
      {
        // replace word symbol with its index (counter)
        // https://github.com/oneapi-src/oneDPL/issues/840
        oneapi::dpl::fill(policy, iter, iter + 1, counter);
        iter = oneapi::dpl::find(policy, iter + 1, d_word_end, h_list[counter]);
      }
    }
  }
  //std::copy(policy, d_word_beg, d_word_end, h_word.begin());
  return h_word;
}


int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    printf("Usage: %s <string length> <repeat>\n", argv[0]);
    exit(1);
  }

  const size_t len = atol(argv[1]);
  const int repeat = atoi(argv[2]);
  const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  std::vector<char> word(len);

  srand(123);
  for (size_t i = 0; i < len; i++) word[i] = a[rand() % 52];

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  auto d_result = mtf(q, word);
  auto h_result = reference(word);
  bool ok = d_result == h_result;
  if (ok) {
    printf("PASS\n");
  }
  else {
    printf("FAIL\n");

    exit(1);
    // output MTF result
    if (len < 16) {
      printf("host: ");
      for (size_t i = 0; i < len; i++)
        printf("%d ", h_result[i]);
      printf("\ndevice: ");
      for (size_t i = 0; i < len; i++)
        printf("%d ", d_result[i]);
      printf("\n");
    }
    return 1;
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) mtf(q, word);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time: " << (time * 1e-9f) / repeat << " (s)\n";
  return 0;
}

