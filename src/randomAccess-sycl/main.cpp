#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

typedef unsigned long long int u64Int;
typedef long long int s64Int;

/* Random number generator */
#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L

#define NUPDATE (4 * TableSize)

#define K1_BLOCKSIZE  256
#define K2_BLOCKSIZE  128

u64Int
HPCC_starts(s64Int n)
{
  int i, j;
  u64Int m2[64];
  u64Int temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;

  #pragma unroll
  for (i=0; i<64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
  }

  for (i=62; i>=0; i--)
    if ((n >> i) & 1)
      break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    #pragma unroll
    for (j=0; j<64; j++)
      if ((ran >> j) & 1)
        temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((s64Int) ran < 0 ? POLY : 0);
  }

  return ran;
}


int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int failure;
  u64Int i;
  u64Int temp;
  //double cputime;               /* CPU time to update table */
  //double realtime;              /* Real time to update table */
  double totalMem;
  u64Int *Table = NULL;
  u64Int logTableSize, TableSize;

  /* calculate local memory per node for the update table */
  totalMem = 1024*1024*512;
  totalMem /= sizeof(u64Int);

  /* calculate the size of update array (must be a power of 2) */
  for (totalMem *= 0.5, logTableSize = 0, TableSize = 1;
       totalMem >= 1.0;
       totalMem *= 0.5, logTableSize++, TableSize <<= 1)
    ; /* EMPTY */

   printf("Table size = %llu\n",  TableSize);

   posix_memalign((void**)&Table, 1024, TableSize * sizeof(u64Int));

  if (! Table ) {
    fprintf( stderr, "Failed to allocate memory for the update table %llu\n", TableSize);
    return 1;
  }

  /* Print parameters for run */
  fprintf( stdout, "Main table size   = 2^%llu = %llu words\n", logTableSize,TableSize);
  fprintf( stdout, "Number of updates = %llu\n", NUPDATE);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  u64Int *d_Table = sycl::malloc_device<u64Int>(TableSize, q);

  sycl::range<1> gws1 ((TableSize+K1_BLOCKSIZE-1) / K1_BLOCKSIZE * K1_BLOCKSIZE);
  sycl::range<1> lws1 (K1_BLOCKSIZE);

  sycl::range<1> gws2 (K2_BLOCKSIZE);
  sycl::range<1> lws2 (K2_BLOCKSIZE);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    /* initialize the table */
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class init_table>(
        sycl::nd_range<1>(gws1, lws1), [=](sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < TableSize) d_Table[i] = i;
      });
    });

    /* update the table */
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class update>(
        sycl::nd_range<1>(gws2, lws2), [=](sycl::nd_item<1> item) {
        int j = item.get_global_id(0);
        u64Int ran = HPCC_starts ((NUPDATE/128) * j);
        for (u64Int i=0; i<NUPDATE/128; i++) {
          ran = (ran << 1) ^ ((s64Int) ran < 0 ? POLY : 0);
          auto atm = sycl::atomic_ref<u64Int,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(d_Table[ran & (TableSize-1)]);
          atm.fetch_xor(ran);
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(Table, d_Table, TableSize * sizeof(u64Int)).wait();

  /* validation */
  temp = 0x1;
  for (i=0; i<NUPDATE; i++) {
    temp = (temp << 1) ^ (((s64Int) temp < 0) ? POLY : 0);
    Table[temp & (TableSize-1)] ^= temp;
  }

  temp = 0;
  for (i=0; i<TableSize; i++)
    if (Table[i] != i) {
      temp++;
    }

  fprintf(stdout, "Found %llu errors in %llu locations (%s).\n",
          temp, TableSize, (temp <= 0.01*TableSize) ? "PASS" : "FAIL");
  if (temp > 0.01*TableSize) exit(1);
  if (temp <= 0.01*TableSize) failure = 0;
  else failure = 1;

  free( Table );
  sycl::free(d_Table, q);
  return failure;
}
