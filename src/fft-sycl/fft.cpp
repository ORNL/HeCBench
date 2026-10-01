#include <cmath>
#include <sycl/sycl.hpp>

#ifdef SINGLE_PRECISION
#define T float
#define T2 sycl::float2
#else
#define T double
#define T2 sycl::double2
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440f
#endif

#define FFT_INLINE inline __attribute__((always_inline))

#define exp_1_8 (T2){1, -1}
#define exp_1_4 (T2){0, -1}
#define exp_3_8 (T2){-1, -1}

#define iexp_1_8 (T2){1, 1}
#define iexp_1_4 (T2){0, 1}
#define iexp_3_8 (T2){-1, 1}

FFT_INLINE T2 exp_i(T phi) {
  return (T2){sycl::cos(phi), sycl::sin(phi)};
}

FFT_INLINE T2 cmplx_mul(T2 a, T2 b) {
  return (T2){a.x() * b.x() - a.y() * b.y(),
              a.x() * b.y() + a.y() * b.x()};
}

FFT_INLINE T2 cm_fl_mul(T2 a, T b) {
  return (T2){b * a.x(), b * a.y()};
}

FFT_INLINE T2 cmplx_add(T2 a, T2 b) {
  return (T2){a.x() + b.x(), a.y() + b.y()};
}

FFT_INLINE T2 cmplx_sub(T2 a, T2 b) {
  return (T2){a.x() - b.x(), a.y() - b.y()};
}

#define FFT2(a0, a1)                     \
  {                                      \
    T2 c0 = *a0;                         \
    *a0 = cmplx_add(c0, *a1);            \
    *a1 = cmplx_sub(c0, *a1);            \
  }

#define FFT4(a0, a1, a2, a3)             \
  {                                      \
    FFT2(a0, a2);                        \
    FFT2(a1, a3);                        \
    *a3 = cmplx_mul(*a3, exp_1_4);       \
    FFT2(a0, a1);                        \
    FFT2(a2, a3);                        \
  }

#define FFT8(a)                                                        \
  {                                                                    \
    FFT2(&a[0], &a[4]);                                                \
    FFT2(&a[1], &a[5]);                                                \
    FFT2(&a[2], &a[6]);                                                \
    FFT2(&a[3], &a[7]);                                                \
                                                                       \
    a[5] = cm_fl_mul(cmplx_mul(a[5], exp_1_8), M_SQRT1_2);             \
    a[6] = cmplx_mul(a[6], exp_1_4);                                   \
    a[7] = cm_fl_mul(cmplx_mul(a[7], exp_3_8), M_SQRT1_2);             \
                                                                       \
    FFT4(&a[0], &a[1], &a[2], &a[3]);                                  \
    FFT4(&a[4], &a[5], &a[6], &a[7]);                                  \
  }

#define IFFT2 FFT2

#define IFFT4(a0, a1, a2, a3)            \
  {                                      \
    IFFT2(a0, a2);                       \
    IFFT2(a1, a3);                       \
    *a3 = cmplx_mul(*a3, iexp_1_4);      \
    IFFT2(a0, a1);                       \
    IFFT2(a2, a3);                       \
  }

#define IFFT8(a)                                                       \
  {                                                                    \
    IFFT2(&a[0], &a[4]);                                               \
    IFFT2(&a[1], &a[5]);                                               \
    IFFT2(&a[2], &a[6]);                                               \
    IFFT2(&a[3], &a[7]);                                               \
                                                                       \
    a[5] = cm_fl_mul(cmplx_mul(a[5], iexp_1_8), M_SQRT1_2);            \
    a[6] = cmplx_mul(a[6], iexp_1_4);                                  \
    a[7] = cm_fl_mul(cmplx_mul(a[7], iexp_3_8), M_SQRT1_2);            \
                                                                       \
    IFFT4(&a[0], &a[1], &a[2], &a[3]);                                 \
    IFFT4(&a[4], &a[5], &a[6], &a[7]);                                 \
  }

void submit_fft_base(sycl::queue& q, T2* work, size_t globalsz, size_t localsz) {
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<T, 1> smem(sycl::range<1>(8 * 8 * 9), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(globalsz), sycl::range<1>(localsz)),
        [=](sycl::nd_item<1> item) {
          int tid = item.get_local_id(0);
          int blockIdx = item.get_group(0) * 512 + tid;
          int hi = tid >> 3;
          int lo = tid & 7;
          T2 data[8];
          const int reversed[] = {0, 4, 2, 6, 1, 5, 3, 7};

          for (int i = 0; i < 8; i++) data[i] = work[blockIdx + i * 64];

          FFT8(data);

          for (int j = 1; j < 8; j++) {
            data[j] = cmplx_mul(
                data[j],
                exp_i(((T)-2 * (T)M_PI * reversed[j] / (T)512) * tid));
          }

          for (int i = 0; i < 8; i++)
            smem[hi * 8 + lo + i * 66] = data[reversed[i]].x();
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            data[i].x() = smem[lo * 66 + hi + i * 8];
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            smem[hi * 8 + lo + i * 66] = data[reversed[i]].y();
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            data[i].y() = smem[lo * 66 + hi + i * 8];
          sycl::group_barrier(item.get_group());

          FFT8(data);

          for (int j = 1; j < 8; j++) {
            data[j] = cmplx_mul(
                data[j],
                exp_i(((T)-2 * (T)M_PI * reversed[j] / (T)64) * hi));
          }

          for (int i = 0; i < 8; i++)
            smem[hi * 8 + lo + i * 72] = data[reversed[i]].x();
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            data[i].x() = smem[hi * 72 + lo + i * 8];
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            smem[hi * 8 + lo + i * 72] = data[reversed[i]].y();
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            data[i].y() = smem[hi * 72 + lo + i * 8];

          FFT8(data);

          for (int i = 0; i < 8; i++)
            work[blockIdx + i * 64] = data[reversed[i]];
        });
  });
}

void submit_ifft_base(sycl::queue& q, T2* work, size_t globalsz, size_t localsz) {
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<T, 1> smem(sycl::range<1>(8 * 8 * 9), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(globalsz), sycl::range<1>(localsz)),
        [=](sycl::nd_item<1> item) {
          int tid = item.get_local_id(0);
          int blockIdx = item.get_group(0) * 512 + tid;
          int hi = tid >> 3;
          int lo = tid & 7;
          T2 data[8];
          const int reversed[] = {0, 4, 2, 6, 1, 5, 3, 7};

          for (int i = 0; i < 8; i++) data[i] = work[blockIdx + i * 64];

          IFFT8(data);

          for (int j = 1; j < 8; j++)
            data[j] = cmplx_mul(
                data[j],
                exp_i(((T)2 * (T)M_PI * reversed[j] / (T)512) * tid));

          for (int i = 0; i < 8; i++)
            smem[hi * 8 + lo + i * 66] = data[reversed[i]].x();
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            data[i].x() = smem[lo * 66 + hi + i * 8];
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            smem[hi * 8 + lo + i * 66] = data[reversed[i]].y();
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            data[i].y() = smem[lo * 66 + hi + i * 8];
          sycl::group_barrier(item.get_group());

          IFFT8(data);

          for (int j = 1; j < 8; j++)
            data[j] = cmplx_mul(
                data[j],
                exp_i(((T)2 * (T)M_PI * reversed[j] / (T)64) * hi));

          for (int i = 0; i < 8; i++)
            smem[hi * 8 + lo + i * 72] = data[reversed[i]].x();
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            data[i].x() = smem[hi * 72 + lo + i * 8];
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            smem[hi * 8 + lo + i * 72] = data[reversed[i]].y();
          sycl::group_barrier(item.get_group());
          for (int i = 0; i < 8; i++)
            data[i].y() = smem[hi * 72 + lo + i * 8];

          IFFT8(data);

          for (int i = 0; i < 8; i++) {
            data[i].x() = data[i].x() / (T)512;
            data[i].y() = data[i].y() / (T)512;
          }

          for (int i = 0; i < 8; i++)
            work[blockIdx + i * 64] = data[reversed[i]];
        });
  });
}
