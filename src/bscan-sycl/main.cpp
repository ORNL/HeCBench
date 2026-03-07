//-----------------------------------------------------------------------
// Reference
//
// Harris, M. and Garland, M., 2012.
// Optimizing parallel prefix operations for the Fermi architecture.
// In GPU Computing Gems Jade Edition (pp. 29-38). Morgan Kaufmann.
//-----------------------------------------------------------------------

#include <cstdio>
#include <chrono>
#include <sycl/sycl.hpp>
#include <cstdlib>

inline int warp_scan(sycl::nd_item<1> &item, int val, volatile int *s_data)
{
  // initialize shared memory accessed by each warp with zeros
  auto sg = item.get_sub_group();
  int warpSize = sg.get_local_linear_range();

  int idx = 2 * item.get_local_id(0) - (item.get_local_id(0) & (warpSize-1));
  s_data[idx] = 0;
  idx += warpSize;
  int t = s_data[idx] = val;
  s_data[idx] = t += s_data[idx - 1];
  s_data[idx] = t += s_data[idx - 2];
  s_data[idx] = t += s_data[idx - 4];
  s_data[idx] = t += s_data[idx - 8];
  s_data[idx] = t += s_data[idx -16];
  return s_data[idx-1];
}

inline unsigned int lanemask_lt(sycl::sub_group &sg)
{
  const unsigned int lane = sg.get_local_linear_id();
  return (1 << (lane)) - 1;
}

// warp scan optimized for binary
inline unsigned int binary_warp_scan(sycl::sub_group &sg, bool p)
{
  const unsigned int mask = lanemask_lt(sg);
  unsigned int b;
  auto gb = sycl::ext::oneapi::group_ballot(sg, p);
  gb.extract_bits(b, 0);
  return sycl::popcount(b & mask);
}

// positive numbers
inline bool valid(int x) {
  return x > 0;
}

inline int block_binary_prefix_sums(sycl::nd_item<1> &item, int *sdata, int x)
{
  bool predicate = valid(x);

  int idx = item.get_local_id(0);
  auto sg = item.get_sub_group();
  int warpSize = sg.get_local_linear_range();

  // A. Compute exclusive prefix sums within each warp
  int warpPrefix = binary_warp_scan(sg, predicate);
  int warpIdx = idx / warpSize;
  int laneIdx = idx & (warpSize - 1);
#ifdef DEBUG
  printf("A %d %d %d\n", warpIdx, laneIdx, warpPrefix);
#endif

  // B. The last thread of each warp stores inclusive
  // prefix sum to the warp’s index in shared memory
  if (laneIdx == warpSize - 1) {
    sdata[warpIdx] = warpPrefix + predicate;
#ifdef DEBUG
    printf("B %d %d\n", warpIdx, sdata[warpIdx]);
#endif
  }
  item.barrier(sycl::access::fence_space::local_space);

  // C. One warp scans the warp partial sums
  if (idx < warpSize) {
    sdata[idx] = warp_scan(item, sdata[idx], sdata);
#ifdef DEBUG
    printf("C: %d %d\n", idx, sdata[idx]);
#endif
  }
  item.barrier(sycl::access::fence_space::local_space);

  // D. Each thread adds prefix sums of warp partial
  // sums to its own intra−warp prefix sums
  return warpPrefix + sdata[warpIdx];
}

void binary_scan(
        sycl::nd_item<1> &item,
        int *__restrict__ s_data,
        int *__restrict__ g_odata,
  const int *__restrict__ g_idata)
{
  int i = item.get_local_id(0);
  g_odata[i] = block_binary_prefix_sums(item, s_data, g_idata[i]);
}

template <int N>
void bscan (const int repeat) 
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int h_in[N];
  int h_out[N];
  int ref_out[N];

  int *d_in = sycl::malloc_device<int>(N, q);
  int *d_out = sycl::malloc_device<int>(N, q);

  bool ok = true;
  double time = 0.0;
  srand(123);

  size_t grid_size = 12*7*8*9*10;
  sycl::range<1> gws (grid_size * N);
  sycl::range<1> lws (N);

  int valid_count = 0;

  for (int i = 0; i < repeat; i++) {
    for (int n = 0; n < N; n++) {
      h_in[n] = rand() % N - N/2;
      if (valid(h_in[n])) valid_count++;  // total number of valid elements
    }
    q.memcpy(d_in, h_in, N*sizeof(int));

    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<int, 1> sm (sycl::range<1>(64), cgh);
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item)
        [[sycl::reqd_sub_group_size(32)]] {
        binary_scan(item, sm.get_multi_ptr<sycl::access::decorated::no>().get(),
                    d_out, d_in);
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // verify exclusive sum
    q.memcpy(h_out, d_out, N*sizeof(int)).wait();

    ref_out[0] = 0;
    ok &= (h_out[0] == ref_out[0]);
    for (int i = 1; i < N; i++) {
      ref_out[i] = ref_out[i-1] + (h_in[i-1] > 0);
      ok &= (ref_out[i] == h_out[i]);
    }
    if (!ok) break;
  } // for

  printf("Block size = %d, ratio of valid elements = %f, verify = %s\n",
          N, valid_count * 1.f / (N * repeat), ok ? "PASS" : "FAIL");

  if (ok) {
    printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);
    printf("Billion elements per second: %f\n\n",
            grid_size * N * repeat / time);
  }

  sycl::free(d_in, q);
  sycl::free(d_out, q);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // scan over N elements (N = [32, 1024])
  bscan<32>(repeat);
  bscan<64>(repeat);
  bscan<128>(repeat);
  bscan<256>(repeat);
  bscan<512>(repeat);
  bscan<1024>(repeat);

  if (!ok) return 1;
  return 0; 
}
