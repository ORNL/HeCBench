#include <stdio.h>
#include <chrono>
#include "utils.h"

static __device__ int currpos1 = 0;
static __device__ int currpos2 = 0;
static __device__ int wlsize = 0;

template <int WarpsPerBlock, int PinLimit>
static __device__
void buildMST(const ID num, 
              const ctype* const __restrict__ x,
              const ctype* const __restrict__ y,
              edge* const __restrict__ edges,
              ctype dist[PinLimit])
{
  const int WS = warpSize;
  __shared__ ID source[WarpsPerBlock][PinLimit];
  __shared__ ID destin[WarpsPerBlock][PinLimit];
  __shared__ ctype mindj[WarpsPerBlock];

  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;

  // initialize
  ID numItems = num - 1;
  for (ID i = lane; i < numItems; i += WS) dist[i] = INT_MAX;  // change if ctype changed
  for (ID i = lane; i < numItems; i += WS) destin[warp][i] = (ID)(i + 1);

  // Prim's MST algorithm
  ID src = 0;
  for (ID cnt = 0; cnt < num - 1; cnt++) {
    if (lane == 0) mindj[warp] = INT_MAX;

    // update distances
    for (ID j = lane; j < numItems; j += WS) {
      const ID dst = destin[warp][j];
      const ctype dnew = abs(x[src] - x[dst]) + abs(y[src] - y[dst]);
      ctype d = dist[j];
      if (d > dnew) {
        d = dnew;
        dist[j] = dnew;
        source[warp][j] = src;
      }
      const int upv = d * (MaxPins * 2) + j;  // tie breaker for determinism
      atomicMin((ctype*)&mindj[warp], upv);
    }

    // create new edge
    const ID j = mindj[warp] % (MaxPins * 2);
    src = destin[warp][j];
    numItems--;
    if (lane == 0) {
      edges[cnt].src = source[warp][j];
      edges[cnt].dst = src;
      dist[j] = dist[numItems];
      source[warp][j] = source[warp][numItems];
      destin[warp][j] = destin[warp][numItems];
    }
  }
}

template <int WarpsPerBlock, int PinLimit>
static __device__
bool insertSteinerPoints(ID& num,
                         ctype* const __restrict__ x,
                         ctype* const __restrict__ y,
                         const edge* const __restrict__ edges,
                         ctype dist[PinLimit])
{
  const int WS = warpSize;
  __shared__ ID adj[WarpsPerBlock][PinLimit][8];
  __shared__ int cnt[WarpsPerBlock][PinLimit];

  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;
  const ID top = num;

  // create adjacency lists
  for (ID i = lane; i < top; i += WS) cnt[warp][i] = 0;
  for (ID e = lane; e < top - 1; e += WS) {
    dist[e] = -1;
    const ID s = edges[e].src;
    const ID d = edges[e].dst;
    if ((x[d] != x[s]) || (y[d] != y[s])) {
      const int ps = atomicAdd(&cnt[warp][s], 1);
      adj[warp][s][ps] = e;
      const int pd = atomicAdd(&cnt[warp][d], 1);
      adj[warp][d][pd] = e;
    }
  }

  // find best distance for each triangle
  for (ID s = lane; s < top; s += WS) {
    if (cnt[warp][s] >= 2) {
      const ctype x0 = x[s];
      const ctype y0 = y[s];
      for (char j = 0; j < cnt[warp][s] - 1; j++) {
        const ID e1 = adj[warp][s][j];
        const ID d1 = (s != edges[e1].src) ? edges[e1].src : edges[e1].dst;
        const ctype x1 = x[d1];
        const ctype y1 = y[d1];
        for (char k = j + 1; k < cnt[warp][s]; k++) {
          const ID e2 = adj[warp][s][k];
          const ID d2 = (s != edges[e2].src) ? edges[e2].src : edges[e2].dst;
          const ctype stx = max(min(x0, x1), min(max(x0, x1), x[d2]));
          const ctype sty = max(min(y0, y1), min(max(y0, y1), y[d2]));
          const ctype rd = abs(stx - x0) + abs(sty - y0);
          if (rd > 0) {
            const ctype rd1 = rd * (MaxPins * 2) + e1;  // tie breaker
            const ctype rd2 = rd * (MaxPins * 2) + e2;  // tie breaker
            atomicMax((ctype*)&dist[e1], rd2);
            atomicMax((ctype*)&dist[e2], rd1);
          }
        }
      }
    }
  }

  // process "triangles" to find best candidate Steiner points
  bool updated = false;
  for (ID e1 = lane; __any(e1 < top - 2); e1 += WS) {
    bool insert = false;
    ctype stx, sty;
    if (e1 < top - 2) {
      const ctype d1 = dist[e1];
      if (d1 > 0) {
        const ID e2 = d1 % (MaxPins * 2);
        if (e2 > e1) {
          const ctype d2 = dist[e2];
          if (e1 == d2 % (MaxPins * 2)) {
            const ctype x0 = x[edges[e1].src];
            const ctype y0 = y[edges[e1].src];
            const ctype x1 = x[edges[e1].dst];
            const ctype y1 = y[edges[e1].dst];
            ctype x2 = x[edges[e2].src];
            ctype y2 = y[edges[e2].src];
            if (((x2 == x0) && (y2 == y0)) || ((x2 == x1) && (y2 == y1))) {
              x2 = x[edges[e2].dst];
              y2 = y[edges[e2].dst];
            }
            updated = true;
            insert = true;
            stx = max(min(x0, x1), min(max(x0, x1), x2));
            sty = max(min(y0, y1), min(max(y0, y1), y2));
          }
        }
      }
    }
    const unsigned long long bal = __ballot(insert);
    const int pos = __popcll(bal & ~(-1ULL << lane)) + num;
    if (insert) {
      x[pos] = stx;
      y[pos] = sty;
    }
    num += __popcll(bal);
  }

  return __any(updated);
}

template <int WarpsPerBlock, int PinLimit>
static __device__
inline void processSmallNet(const int i,
                            const int* const __restrict__ idxin,
                            const ctype* const __restrict__ xin,
                            const ctype* const __restrict__ yin,
                            int* const __restrict__ idxout,
                            ctype* const __restrict__ xout,
                            ctype* const __restrict__ yout,
                             edge* const __restrict__ edges,
                              int* const __restrict__ wl)
{
  const int WS = warpSize;
  __shared__ ctype dist[WarpsPerBlock][PinLimit];
  const int lane = threadIdx.x % WS;
  const int warp = threadIdx.x / WS;

  // initialize arrays and copy input coords to output
  const int pin = idxin[i];
  const ID num = idxin[i + 1] - pin;
  const int pout = 2 * pin;
  if (lane == 0) idxout[i] = pout;
  for (ID j = lane; j < num; j += WS) xout[pout + j] = xin[pin + j];
  for (ID j = lane; j < num; j += WS) yout[pout + j] = yin[pin + j];

  // process nets
  if (num == 2) {
    if (lane == 0) edges[pout] = edge{0, 1};
  } else if (num == 3) {
    ctype x0, y0;
    if (lane < 3) {
      edges[pout + lane] = edge{(short)lane, 3};
      x0 = xout[pout + lane];
      y0 = yout[pout + lane];
    }
    const ctype x1 = __shfl(x0, 1);
    const ctype y1 = __shfl(y0, 1);
    const ctype x2 = __shfl(x0, 2);
    const ctype y2 = __shfl(y0, 2);
    if (lane == 0) {
      xout[pout + 3] = max(min(x0, x1), min(max(x0, x1), x2));
      yout[pout + 3] = max(min(y0, y1), min(max(y0, y1), y2));
    }
  } else if (num <= 32) {
    // iterate until all Steiner points added
    ID cnt = num;
    do {
      buildMST<WarpsPerBlock, PinLimit>(cnt, &xout[pout], &yout[pout], &edges[pout], dist[warp]);
    } while (insertSteinerPoints<WarpsPerBlock, PinLimit>(cnt, &xout[pout], &yout[pout], &edges[pout], dist[warp]));
  } else {
    if (lane == 0) wl[atomicAdd(&wlsize, 1)] = i;
  }
}

template <int WarpsPerBlock, int PinLimit>
static __device__
inline void processLargeNet(const int i,
                            const int* const __restrict__ idxin,
                            ctype* const __restrict__ xout,
                            ctype* const __restrict__ yout,
                             edge* const __restrict__ edges)
{
  const int WS = warpSize;
  __shared__ ctype dist[WarpsPerBlock][PinLimit];
  const int warp = threadIdx.x / WS;

  const int pin = idxin[i];
  const ID num = idxin[i + 1] - pin;
  const int pout = 2 * pin;

  // iterate until all Steiner points added
  ID cnt = num;
  do {
    buildMST<WarpsPerBlock, PinLimit>(cnt, &xout[pout], &yout[pout], &edges[pout], dist[warp]);
  } while (insertSteinerPoints<WarpsPerBlock, PinLimit>(cnt, &xout[pout], &yout[pout], &edges[pout], dist[warp]));
}

template <int WarpsPerBlock, int PinLimit, int WaveSize>
static __global__ __launch_bounds__(WarpsPerBlock * WaveSize, 2)
void largeNetKernel(const int* const __restrict__ idxin,
                    const ctype* const __restrict__ xin,
                    const ctype* const __restrict__ yin,
                    int* const __restrict__ idxout,
                    ctype* __restrict__ xout,
                    ctype* __restrict__ yout,
                     edge* __restrict__ edges,
                    const int numnets,
                    int* const __restrict__ wl)
{
  const int WS = warpSize;
  // compute Steiner points and edges
  const int lane = threadIdx.x % WS;
  do {
    int i;
    if (lane == 0) i = atomicAdd(&currpos1, 1);
    i = __shfl(i, 0);
    if (i >= numnets) break;
    processSmallNet<WarpsPerBlock, PinLimit>(i, idxin, xin, yin, idxout, xout, yout, edges, wl);
  } while (true);

  // set final element
  if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
    idxout[numnets] = 2 * idxin[numnets];
  }
}

template <int WarpsPerBlock, int PinLimit, int WaveSize>
static __global__ __launch_bounds__(WarpsPerBlock * WaveSize, 2)
void smallNetKernel(const int* const __restrict__ idxin,
                    ctype* __restrict__ xout,
                    ctype* __restrict__ yout,
                     edge* __restrict__ edges,
                      int* const __restrict__ wl)
{
  const int WS = warpSize;
  // compute Steiner points and edges
  const int lane = threadIdx.x % WS;
  do {
    int i;
    if (lane == 0) i = atomicAdd(&currpos2, 1);
    i = __shfl(i, 0);
    if (i >= wlsize) break;
    processLargeNet<WarpsPerBlock, PinLimit>(wl[i], idxin, xout, yout, edges);
  } while (true);
}

static void computeRSMT(const int* const __restrict__ idxin,
                        const ctype* const __restrict__ xin,
                        const ctype* const __restrict__ yin,
                         int* const __restrict__ idxout,
                       ctype* const __restrict__ xout,
                       ctype* const __restrict__ yout,
                        edge* const __restrict__ edges,
                        const int numnets)
{
  // obtain GPU info
  int WarpSize;
  GPU_CHECK(hipDeviceGetAttribute(&WarpSize, hipDeviceAttributeWarpSize, 0));

  hipDeviceProp_t deviceProp;
  GPU_CHECK(hipGetDeviceProperties(&deviceProp, 0));
  const int SMs = deviceProp.multiProcessorCount;
  const int blocks = SMs * 2;
  const int largeWarpsPerBlock = (WarpSize == 64) ? 12 : 24;
  printf("launching %d thread blocks with %d threads per block (warp size %d)\n",
         blocks, largeWarpsPerBlock * WarpSize, WarpSize);

  // allocate and initialize GPU memory
  int* d_idxin;  ctype* d_xin;  ctype* d_yin;
  int* d_idxout;  ctype* d_xout;  ctype* d_yout;  edge* d_edges;
  int* d_wl;
  const int size = idxin[numnets];
  GPU_CHECK(hipMalloc((void **)&d_idxin, (numnets + 1) * sizeof(int)));
  GPU_CHECK(hipMalloc((void **)&d_xin, size * sizeof(ctype)));
  GPU_CHECK(hipMalloc((void **)&d_yin, size * sizeof(ctype)));
  GPU_CHECK(hipMalloc((void **)&d_idxout, (numnets + 1) * sizeof(int)));
  GPU_CHECK(hipMalloc((void **)&d_xout, 2 * size * sizeof(ctype)));
  GPU_CHECK(hipMalloc((void **)&d_yout, 2 * size * sizeof(ctype)));
  GPU_CHECK(hipMalloc((void **)&d_edges, 2 * size * sizeof(edge)));
  GPU_CHECK(hipMalloc((void **)&d_wl, numnets * sizeof(int)));
  GPU_CHECK(hipMemcpy(d_idxin, idxin, (numnets + 1) * sizeof(int), hipMemcpyHostToDevice));
  GPU_CHECK(hipMemcpy(d_xin, xin, size * sizeof(ctype), hipMemcpyHostToDevice));
  GPU_CHECK(hipMemcpy(d_yin, yin, size * sizeof(ctype), hipMemcpyHostToDevice));

  // start time
  GPU_CHECK(hipDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  // process nets
  GPU_CHECK(hipMemset(d_xout, -1, 2 * size * sizeof(ctype)));
  GPU_CHECK(hipMemset(d_yout, -1, 2 * size * sizeof(ctype)));
  GPU_CHECK(hipMemset(d_edges, 0, 2 * size * sizeof(edge)));

  if (WarpSize == 64) {
    largeNetKernel<12, 64, 64><<<blocks, 12 * WarpSize>>>(
      d_idxin, d_xin, d_yin, d_idxout, d_xout, d_yout, d_edges, numnets, d_wl);
    smallNetKernel<3, 512, 64><<<blocks, 3 * WarpSize>>>(
      d_idxin, d_xout, d_yout, d_edges, d_wl);
  } else {
    largeNetKernel<24, 64, 32><<<blocks, 24 * WarpSize>>>(
      d_idxin, d_xin, d_yin, d_idxout, d_xout, d_yout, d_edges, numnets, d_wl);
    smallNetKernel<3, 512, 32><<<blocks, 3 * WarpSize>>>(
      d_idxin, d_xout, d_yout, d_edges, d_wl);
  }

  // end time
  GPU_CHECK(hipDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  const double runtime = time * 1e-9;
  printf("compute time: %.6f s\n", runtime);
  printf("throughput: %.f nets/sec\n", numnets / runtime);

  // transfer results from GPU
  GPU_CHECK(hipMemcpy(idxout, d_idxout, (numnets + 1) * sizeof(int), hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(xout, d_xout, 2 * size * sizeof(ctype), hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(yout, d_yout, 2 * size * sizeof(ctype), hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(edges, d_edges, 2 * size * sizeof(edge), hipMemcpyDeviceToHost));

  // clean up
  GPU_CHECK(hipFree(d_wl));
  GPU_CHECK(hipFree(d_edges));
  GPU_CHECK(hipFree(d_yout));
  GPU_CHECK(hipFree(d_xout));
  GPU_CHECK(hipFree(d_idxout));
  GPU_CHECK(hipFree(d_yin));
  GPU_CHECK(hipFree(d_xin));
  GPU_CHECK(hipFree(d_idxin));
}

int main(int argc, char* argv[])
{
  printf("A Simple, Fast, and GPU-friendly Steiner-Tree Heuristic\n");
  printf("Copyright 2019-2022 Texas State University\n\n");

  if (argc != 2) {
    printf("Usage: %s file_name \n", argv[0]);
    exit(-1);
  }

  printf("reading input file: %s\n", argv[1]);
  grid g;
  net_list n;
  read_file(argv[1], g, n);
  const int numnets = n.num_net;

  int* idxin = NULL;
  GPU_CHECK(hipHostAlloc(&idxin, (numnets + 1) * sizeof(int), hipHostMallocDefault));

  // initialize idxin
  idxin[0] = 0;
  ID hipin = 0;
  int pos = 0;
  for (int i = 0; i < numnets; i++) {
    const ID num = std::min(n.net_num_pins[i], MaxPins);
    hipin = std::max(hipin, num);
    pos += num;
    idxin[i + 1] = pos;
  }

  // histogram and trucated nets
  int trunc = 0;
  int* const hist = new int [hipin + 1];
  for (int i = 0; i < hipin + 1; i++) hist[i] = 0;
  for (int i = 0; i < numnets; i++) {
    const ID num = std::min(n.net_num_pins[i], MaxPins);
    if (num < n.net_num_pins[i]) trunc++;
    hist[num]++;
  }
  int sum = 0;
  for (int i = 0; i < hipin + 1; i++) {
    sum += hist[i];
  }
  delete [] hist;

  printf("number of nets: %d\n", numnets);
  printf("max pins per net: %d\n", hipin);
  printf("truncated nets: %d\n", trunc);
  if (hipin > MaxPins) {printf("ERROR: hi_pin_count must be no more than %d\n", MaxPins); exit(-1);}

  // pin coordinates
  ctype* xin = NULL;
  GPU_CHECK(hipHostAlloc(&xin, idxin[numnets] * sizeof(ctype), hipHostMallocDefault));

  ctype* yin = NULL;
  GPU_CHECK(hipHostAlloc(&yin, idxin[numnets] * sizeof(ctype), hipHostMallocDefault));

  // initialize pin coordinates
  pos = 0;
  for (int i = 0; i < numnets; i++) {
    const ID num = idxin[i + 1] - idxin[i];
    for (ID j = 0; j < num; j++) xin[pos + j] = std::get<0>(n.num_net_arr[i][j]);
    for (ID j = 0; j < num; j++) yin[pos + j] = std::get<1>(n.num_net_arr[i][j]);
    pos += num;
  }

  // result storage
  const int size = 2 * idxin[numnets];
  int* idxout = NULL;
  GPU_CHECK(hipHostAlloc(&idxout, (numnets + 1) * sizeof(int), hipHostMallocDefault));

  ctype* xout = NULL;
  GPU_CHECK(hipHostAlloc(&xout, size * sizeof(ctype), hipHostMallocDefault));

  ctype* yout = NULL;
  GPU_CHECK(hipHostAlloc(&yout, size * sizeof(ctype), hipHostMallocDefault));

  edge* edges = NULL;
  GPU_CHECK(hipHostAlloc(&edges, size * sizeof(edge), hipHostMallocDefault));

  // compute Steiner points and edges
  computeRSMT(idxin, xin, yin, idxout, xout, yout, edges, numnets);

  // print results
  long total_len = 0, total_pin = 0;
  
  for (int i = 0; i < numnets; i++) {
    // body of treeLength function illustrates how to read solution
    const ctype len = treeLength(idxout[i + 1] - idxout[i],
                      &xout[idxout[i]], &yout[idxout[i]], &edges[idxout[i]]);
    const ID num = std::min(n.net_num_pins[i], MaxPins);
    total_len += len;
    total_pin += num;
  }
  printf("total wirelength: %ld\n", total_len);
  printf("total pins: %ld\n", total_pin);

  // clean up
  free_memory(g, n);
  GPU_CHECK(hipHostFree(edges));
  GPU_CHECK(hipHostFree(yout));
  GPU_CHECK(hipHostFree(xout));
  GPU_CHECK(hipHostFree(idxout));
  GPU_CHECK(hipHostFree(yin));
  GPU_CHECK(hipHostFree(xin));
  GPU_CHECK(hipHostFree(idxin));

  return 0;
}
