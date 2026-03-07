#include <stdio.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>

#define P1 55
#define P2 119
#define P3 179
#define P4 256

#define LWDR 32
#define LKNB 8

typedef unsigned int uint32_t;

#define __syncthreads() item.barrier(sycl::access::fence_space::local_space)

void LFIB4(uint32_t n, uint32_t *x) {
  #pragma omp simd safelen(32)
  for (uint32_t k = P4; k < n; k++) {
    x[k] = x[k - P1] + x[k - P2] + x[k - P3] + x[k - P4];
  }
}

void firstColGPU(uint32_t *x, int s, sycl::nd_item<1> &item, uint32_t *cx) {

  uint32_t *px = &cx[P4];
  int myid = item.get_local_id(0);
  cx[myid] = x[myid];

  __syncthreads();

  for (int k = 1; k < s / P4; k++) {

    for (int i = 0; i < P4; i += LWDR) {
      if (myid < LWDR) {
        px[i + myid] = px[i + myid - P1] + px[i + myid - P2]
                     + px[i + myid - P3] + px[i + myid - P4];
      }
      __syncthreads();
    }

    x[k * P4 + myid] = cx[myid] = px[myid];
    __syncthreads();
  }
}

void colYGPU(uint32_t *y, int s, sycl::nd_item<1> &item, uint32_t *cy) {

  uint32_t *ay = &cy[P4 * 2];
  int myid = item.get_local_id(0);
  ay[myid] = y[2 * P4 + myid];
  __syncthreads();

  for (int k = 0; k < s / P4; k++) {

    cy[myid] = cy[myid + P4];
    cy[myid + P4] = ay[myid];
    __syncthreads();

    for (int i = 0; i < P4; i += LWDR) {
      if (myid < LWDR) {
        ay[i + myid] = ay[i + myid - P1] + ay[i + myid - P2]
                     + ay[i + myid - P3] + ay[i + myid - P4];
      }
      __syncthreads();
    }
  }

  y[2 * P4 + myid] = cy[2 * P4 + myid];
  y[P4 + myid] = cy[P4 + myid];
  y[myid] = cy[myid];
}

void lastEntGPU(uint32_t *__restrict x,
                uint32_t *__restrict y,
                int s, int r,
                sycl::nd_item<1> &item,
                uint32_t *__restrict a0,
                uint32_t *__restrict b0,
                uint32_t *__restrict c0,
                uint32_t *__restrict d0) {

  uint32_t *a = a0 + P4;
  uint32_t *b = b0 + P4;
  uint32_t *c = c0 + P4;
  uint32_t *d = d0 + P4;

  int myid = item.get_local_id(0);

  a0[myid] = y[myid];
  __syncthreads();

  if (myid < P4)
    a0[myid + P4 * 2] = y[myid + P4 * 2];
  __syncthreads();

  d0[myid] = c0[myid] = b0[myid] = a[myid];
  __syncthreads();

  b[myid - P4] += a[-(P4 - P3) + myid];
  __syncthreads();

  c[myid - P4] += (a[-(P3 - P2) + myid] + a[-(P4 - P2) + myid]);
  __syncthreads();

  d[myid - P4] += (a[-(P2 - P1) + myid] + a[-(P3 - P1) + myid]
      + a[-(P4 - P1) + myid]);
  __syncthreads();

  a += P4;

  for (int i = 1; i < r; i++) {

    uint32_t *xc = &x[i * s];
    uint32_t tmp = 0;

    if (myid < P4) {

      for (int k = 0; k < P4 - P3; k++)
        tmp += xc[-P4 + k] * a[myid - k];

      for (int k = 0; k < P3 - P2; k++)
        tmp += xc[-P3 + k] * b[myid - k];

      for (int k = 0; k < P2 - P1; k++)
        tmp += xc[-P2 + k] * c[myid - k];

      for (int k = 0; k < P1; k++)
        tmp += xc[-P1 + k] * d[myid - k];

      xc[s - P4 + myid] = tmp;

    }
    __syncthreads();
  }
}

void colsGPU(uint32_t *x, int s, int r, sycl::nd_item<1> &item,
             const sycl::local_accessor<uint32_t, 2> &cx) {
  int lid = item.get_local_id(0);
  int gid = item.get_group(0);
  int dim = item.get_group_range(0);

  int k0 = gid * LKNB;
  int k1 = lid / LWDR;
  int k2 = lid % LWDR;

  int fcol = (gid == 0) ? 1 : 0;
  int ecol = (gid == dim - 1 && r % LKNB) ? r % LKNB : LKNB;

  for (int i = fcol; i < ecol; i++)
    cx[i][lid] = x[(k0 + i) * s - P4 + lid];

  __syncthreads();

  uint32_t *pcx = &cx[k1][P4];

  for (int k = 0; k < s / P4 - 1; k++) {

    for (int i = 0; i < P4; i += LWDR)
    {
      if (!(gid == 0 && lid == 0) && !(gid == dim - 1 && k1 >= ecol))
        pcx[i + k2] = pcx[i + k2 - P1] + pcx[i + k2 - P2]
                    + pcx[i + k2 - P3] + pcx[i + k2 - P4];

      __syncthreads();
    }

    for (int i = fcol; i < ecol; i++)
      x[(k0 + i) * s + k * P4 + lid] = cx[i][lid] = cx[i][P4 + lid];

    __syncthreads();
  }
}

void gLFIB4(sycl::queue &q, uint32_t n, uint32_t *x, int s, int r, uint32_t *seed) {

  auto e_copy_x = q.memcpy(x, seed, sizeof(uint32_t) * P4);

  uint32_t one = 1;

  uint32_t *y = sycl::malloc_device<uint32_t>(3 * P4, q);
  q.memset(y + P4 * 2, 0, P4 * sizeof(uint32_t));
  q.memcpy(y + P4 * 2, &one, sizeof(uint32_t));

  sycl::range<1> gws (P4);
  sycl::range<1> lws (P4);

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint32_t, 1> cx (sycl::range<1>(2 * P4), cgh);
    cgh.parallel_for<class firstCol>(
      sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      firstColGPU(x, s, item, cx.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint32_t, 1> cy (sycl::range<1>(3 * P4), cgh);
    cgh.parallel_for<class colY>(
      sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      colYGPU(y, s, item, cy.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });

  sycl::range<1> gws2 (2*P4);
  sycl::range<1> lws2 (2*P4);

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint32_t, 1> a0 (sycl::range<1>(3 * P4), cgh);
    sycl::local_accessor<uint32_t, 1> b0 (sycl::range<1>(2 * P4), cgh);
    sycl::local_accessor<uint32_t, 1> c0 (sycl::range<1>(2 * P4), cgh);
    sycl::local_accessor<uint32_t, 1> d0 (sycl::range<1>(2 * P4), cgh);
    cgh.parallel_for<class lastEnt>(
      sycl::nd_range<1>(gws2, lws2), [=](sycl::nd_item<1> item) {
      lastEntGPU(x, y, s, r, item,
                 a0.get_multi_ptr<sycl::access::decorated::no>().get(),
                 b0.get_multi_ptr<sycl::access::decorated::no>().get(),
                 c0.get_multi_ptr<sycl::access::decorated::no>().get(),
                 d0.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });

  sycl::range<1> gws3 ((r / LKNB + (r % LKNB ? 1 : 0)) * P4);
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint32_t, 2> cx (sycl::range<2>(LKNB, 2 * P4), cgh);
    cgh.parallel_for<class cols>(
      sycl::nd_range<1>(gws3, lws), [=](sycl::nd_item<1> item) {
      colsGPU(x, s, r, item, cx);
    });
  });

  q.wait();
  sycl::free(y, q);
}

int main(int argc, char **argv) {

  if (argc < 1) {
    printf("Usage: ./main <n>\n");
    return 1;
  }

  uint32_t n = atoi(argv[1]);

  srand(1234);
  uint32_t *x = (uint32_t*) malloc(n * sizeof(uint32_t));

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  for (uint32_t r = 16; r <= 4096; r = r * 2) {

    uint32_t s = 0;

    if (s == 0) {
      s = n / r;
      s -= (s % 256 == 0 ? 0 : s % 256);
      while (s * r < n) r++;
    }

    printf("n=%d r=%d s=%d\n", n, r, s);

    uint32_t *z = (uint32_t*) malloc(r * s * sizeof(uint32_t));

    for (uint32_t k = 0; k < P4; k++) x[k] = z[k] = rand();

    // compute on the host
    auto start = std::chrono::steady_clock::now();
    LFIB4(n, x);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> host_time = end - start;

    // compute on the device
    uint32_t *x_d = sycl::malloc_device<uint32_t>(r * s, q);

    start = std::chrono::steady_clock::now();
    gLFIB4(q, n, x_d, s, r, z);
    end = std::chrono::steady_clock::now();
    std::chrono::duration<float> device_time = end - start;
    printf("r = %d | host time = %lf | device time = %lf | speedup = %.1f ",
      r, host_time.count(), device_time.count(), host_time.count() / device_time.count());

    // verify
    q.memcpy(z, x_d, sizeof(uint32_t) * n).wait();

    bool ok = true;
    for (uint32_t i = 0; i < n; i++) {
      if (x[i] != z[i]) {
        ok = false;
        break;
      }
    }
    printf("check = %s\n", ok ? "PASS" : "FAIL");
    if (!ok) exit(1);

    free(z);
    sycl::free(x_d, q);
  }

  free(x);
  return 0;
}
