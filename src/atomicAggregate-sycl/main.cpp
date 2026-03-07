#include <chrono>
#include <stdio.h>
#include <sycl/sycl.hpp>
#include <cstdlib>

// reference
// https://stackoverflow.com/questions/59879285/whats-the-alternative-for-match-any-sync-on-compute-capability-6

#define warpSize 32

inline int ffs(int x) {
  return (x == 0) ? 0 : sycl::ctz(x) + 1;
}

#define DPCT
/// https://github.com/oneapi-src/SYCLomatic/
/// The function match_any_over_sub_group conducts a comparison of values
/// across work-items within a sub-group. match_any_over_sub_group return a mask
/// in which some bits are set to 1, indicating that the values provided by
/// the work-items represented by these bits are equal. The n-th bit of mask
/// representing the work-item with id n. The parameter \p member_mask
/// indicating the work-items participating the call.
template <typename T>
unsigned int match_any_over_sub_group(sycl::sub_group g, unsigned member_mask,
                                      T value) {
  static_assert(std::is_trivially_copyable_v<T>,
                "Value type must be trivially copyable type.");
  if (!member_mask) {
    return 0;
  }
  unsigned int id = g.get_local_linear_id();
  unsigned int flag = 0, result = 0, reduce_result = 0;
  unsigned int bit_index = 0x1 << id;
  bool is_participate = member_mask & bit_index;
  T broadcast_value = 0;
  bool matched = false;
  while (flag != member_mask) {
    broadcast_value =
        sycl::select_from_group(g, value, sycl::ctz((~flag & member_mask)));
    reduce_result = sycl::reduce_over_group(
        g, is_participate ? (broadcast_value == value ? bit_index : 0) : 0,
        sycl::plus<>());
    flag |= reduce_result;
    matched = reduce_result & bit_index;
    result = matched * reduce_result + (1 - matched) * result;
  }
  return result;
}


// increment the value at ptr by 1 and return the old value
int atomicAggInc(int* ptr, sycl::nd_item<1> &item) {
  int mask;
  auto sg = item.get_sub_group();
#ifdef DPCT
  mask = match_any_over_sub_group(sg, 0xFFFFFFFF, (unsigned long long)ptr);
#else
  for (int i = 0; i < warpSize; i++) {
    unsigned long long tptr = sycl::select_from_group(sg, (unsigned long long)ptr, i);
    auto gb = sycl::ext::oneapi::group_ballot(sg, (tptr == (unsigned long long)ptr));
    unsigned my_mask;
    gb.extract_bits(my_mask, 0);
    if (i == (item.get_local_id(0) & (warpSize - 1))) mask = my_mask;
  }
#endif

  int leader = ffs(mask) - 1; // select a leader
  int res = 0;
  unsigned lane_id = item.get_local_id(0) % warpSize;
  if (lane_id == leader) {                 // leader does the update
    sycl::atomic_ref<int, 
       sycl::memory_order::relaxed,
       sycl::memory_scope::device,
       sycl::access::address_space::global_space> ao (*ptr);
    res = ao.fetch_add(sycl::popcount(mask));
  }
  res = sycl::select_from_group(sg, res, leader); // get leader’s old value
  return res + sycl::popcount(mask & ((1 << lane_id) - 1)); // compute old value
}

void k(int *d, int s, sycl::nd_item<1> &item) {
  int *ptr = d + item.get_local_id(0) % s;
  atomicAggInc(ptr, item);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int nBlocks = 65536;
  const int blockSize = 256;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  for (int ds = 32; ds >= 1; ds = ds / 2) {

    int *d_d, *h_d;
    h_d = new int[ds];

    d_d = (int *)sycl::malloc_device(ds * sizeof(d_d[0]), q);
    q.memset(d_d, 0, ds * sizeof(d_d[0]));

    q.wait();

    sycl::range<1> gws (blockSize * nBlocks);
    sycl::range<1> lws (blockSize);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class kernel>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) 
                         [[sycl::reqd_sub_group_size(warpSize)]] {
          k(d_d, ds, item);
        });
      });
    }

    q.wait();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> time = end - start;
    printf("Total kernel time (%d locations): %f (s)\n", ds, time.count());

    q.memcpy(h_d, d_d, ds * sizeof(d_d[0])).wait();

    bool ok = true;
    for (int i = 0; i < ds; i++) {
      if (h_d[i] != blockSize / ds * nBlocks * repeat) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    sycl::free(d_d, q);
    delete [] h_d;
  }
  if (!ok) return 1;
  return 0;
}
