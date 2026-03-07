#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "utils.h"

__global__
void wyllie ( long *list , const int size )
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size )
  {
    long node, next;
    while ( ((node = list[index]) >> 32) != NIL && 
            ((next = list[node >> 32]) >> 32) != NIL )
    {
      long temp = (node & MASK) ;
      temp += (next & MASK) ;
      temp += (next >> 32) << 32;
      __syncthreads();
      list [ index ] = temp ;
    } 
  }
}

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
  long *d_list;
  hipMalloc((void**)&d_list, sizeof(long) * elems); 

  dim3 grid ((elems + 255)/256);
  dim3 block (256);

  double time = 0.0;

  for (i = 0; i <= repeat; i++) {
    hipMemcpy(d_list, list.data(), sizeof(long) * elems, hipMemcpyHostToDevice);

    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    hipLaunchKernelGGL(wyllie, grid, block, 0, 0, d_list, elems);

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    if (i > 0) time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  hipMemcpy(d_res.data(), d_list, sizeof(long) * elems, hipMemcpyDeviceToHost);
  hipFree(d_list); 

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
    printf("%d: %ld %ld\n", i, h_res[i], d_res[i]);
  }
#endif

  printf("%s\n", (h_res == d_res) ? "PASS" : "FAIL");
  if (!(h_res == d_res)) exit(1);
   
  return 0;
}
