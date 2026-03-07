#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sycl/sycl.hpp>
#include "utils.h"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: ./%s <list size> <0 or 1> <repeat>", argv[0]);
    printf("0 and 1 indicate an ordered list and a random list, respectively\n");
    exit(-1);
  }

  int elems = atoi(argv[1]);
  int setRandomList = atoi(argv[2]);
  int repeat = atoi(argv[3]);
  int i;

  std::vector<int> next (elems);
  std::vector<int> rank (elems);
  std::vector<long> list (elems);
  std::vector<long> d_res (elems);
  std::vector<long> h_res (elems);

  // generate an array in which each element contains the index of the next element
  if (setRandomList)
    random_list(next);
  else
    ordered_list(next);

  // initialize the rank list
  for (i = 0; i < elems; i++) {
    rank[i] = next[i] == NIL ? 0 : 1;
  }

  // pack next and rank as a 64-bit number
  for (i = 0; i < elems; i++) list[i] = ((long)next[i] << 32) | rank[i];

  // run list ranking on a device
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  long *d_list = sycl::malloc_device<long>(elems, q);

  sycl::range<1> gws ((elems + 255)/256*256);
  sycl::range<1> lws (256);

  double time = 0.0;

  for (i = 0; i <= repeat; i++) {
    q.memcpy(d_list, list.data(), sizeof(long) * elems);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class wyllie>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int index = item.get_global_id(0);
        if(index < elems ) {
          long node, next;
          while ( ((node = d_list[index]) >> 32) != NIL &&
                  ((next = d_list[node >> 32]) >> 32) != NIL ) {
            long temp = (node & MASK) ;
            temp += (next & MASK) ;
            temp += (next >> 32) << 32;
            item.barrier(sycl::access::fence_space::local_space);
            d_list [index] = temp ;
          }
        }
      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    if (i > 0) time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  q.memcpy(d_res.data(), d_list, sizeof(long) * elems).wait();
  sycl::free(d_list, q);

  for (i = 0; i < elems; i++) d_res[i] &= MASK;

  // verify
  // compute distance from the *end* of the list (note the first element is the head node)
  h_res[0] = elems-1;
  i = 0;
  for (int r = 1; r < elems; r++) {
    h_res[next[i]] = elems-1-r;
    i = next[i];
  }


#ifdef DEBUG
  printf("Ranks:\n");
  for (i = 0; i < elems; i++) {
    printf("%d: %d %d\n", i, h_res[i], d_res[i]);
  }
#endif

  printf("%s\n", (h_res == d_res) ? "PASS" : "FAIL");
  if (!(h_res == d_res)) exit(1);

  return 0;
}
