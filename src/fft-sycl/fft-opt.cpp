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

#if defined(__SYCL_DEVICE_ONLY__)
typedef T FFT_VEC2 __attribute__((ext_vector_type(2)));
#define FFT_LOAD_SC(dst_x, dst_y, src)                                          \
  do {                                                                         \
    FFT_VEC2 _fft_v = *reinterpret_cast<const FFT_VEC2*>(&(src));              \
    (dst_x) = _fft_v.x;                                                        \
    (dst_y) = _fft_v.y;                                                        \
  } while (0)
#define FFT_STORE_SC(dst, src_x, src_y)                                        \
  do {                                                                         \
    FFT_VEC2 _fft_v = {(T)(src_x), (T)(src_y)};                                \
    *reinterpret_cast<FFT_VEC2*>(&(dst)) = _fft_v;                             \
  } while (0)
#else
#define FFT_LOAD_SC(dst_x, dst_y, src)                                          \
  do {                                                                         \
    (dst_x) = (src).x();                                                       \
    (dst_y) = (src).y();                                                       \
  } while (0)
#define FFT_STORE_SC(dst, src_x, src_y)                                        \
  do {                                                                         \
    (dst).x() = (src_x);                                                       \
    (dst).y() = (src_y);                                                       \
  } while (0)
#endif

#define CMPLX_MUL_SC(ax, ay, bx, by)                                           \
  do {                                                                         \
    T _fft_x = (ax) * (bx) - (ay) * (by);                                      \
    T _fft_y = (ax) * (by) + (ay) * (bx);                                      \
    (ax) = _fft_x;                                                             \
    (ay) = _fft_y;                                                             \
  } while (0)

#define CM_FL_MUL_SC(ax, ay, b)                                                \
  do {                                                                         \
    (ax) = (b) * (ax);                                                         \
    (ay) = (b) * (ay);                                                         \
  } while (0)

#define CMPLX_MUL_EXP_SC(ax, ay, phi)                                          \
  do {                                                                         \
    T _fft_phi = (phi);                                                        \
    T _fft_c = sycl::cos(_fft_phi);                                            \
    T _fft_s = sycl::sin(_fft_phi);                                            \
    CMPLX_MUL_SC(ax, ay, _fft_c, _fft_s);                                      \
  } while (0)

#define FFT2_SC(a0x, a0y, a1x, a1y)                                            \
  do {                                                                         \
    T _fft_x = (a0x);                                                          \
    T _fft_y = (a0y);                                                          \
    (a0x) = _fft_x + (a1x);                                                    \
    (a0y) = _fft_y + (a1y);                                                    \
    (a1x) = _fft_x - (a1x);                                                    \
    (a1y) = _fft_y - (a1y);                                                    \
  } while (0)

#define FFT4_SC(a0x, a0y, a1x, a1y, a2x, a2y, a3x, a3y)                       \
  do {                                                                         \
    FFT2_SC(a0x, a0y, a2x, a2y);                                               \
    FFT2_SC(a1x, a1y, a3x, a3y);                                               \
    CMPLX_MUL_SC(a3x, a3y, (T)0, (T)-1);                                      \
    FFT2_SC(a0x, a0y, a1x, a1y);                                               \
    FFT2_SC(a2x, a2y, a3x, a3y);                                               \
  } while (0)

#define FFT8_SC(a0x, a0y, a1x, a1y, a2x, a2y, a3x, a3y,                       \
                a4x, a4y, a5x, a5y, a6x, a6y, a7x, a7y)                       \
  do {                                                                         \
    FFT2_SC(a0x, a0y, a4x, a4y);                                               \
    FFT2_SC(a1x, a1y, a5x, a5y);                                               \
    FFT2_SC(a2x, a2y, a6x, a6y);                                               \
    FFT2_SC(a3x, a3y, a7x, a7y);                                               \
    CMPLX_MUL_SC(a5x, a5y, (T)1, (T)-1);                                      \
    CM_FL_MUL_SC(a5x, a5y, (T)M_SQRT1_2);                                     \
    CMPLX_MUL_SC(a6x, a6y, (T)0, (T)-1);                                      \
    CMPLX_MUL_SC(a7x, a7y, (T)-1, (T)-1);                                     \
    CM_FL_MUL_SC(a7x, a7y, (T)M_SQRT1_2);                                     \
    FFT4_SC(a0x, a0y, a1x, a1y, a2x, a2y, a3x, a3y);                          \
    FFT4_SC(a4x, a4y, a5x, a5y, a6x, a6y, a7x, a7y);                          \
  } while (0)

#define IFFT4_SC(a0x, a0y, a1x, a1y, a2x, a2y, a3x, a3y)                      \
  do {                                                                         \
    FFT2_SC(a0x, a0y, a2x, a2y);                                               \
    FFT2_SC(a1x, a1y, a3x, a3y);                                               \
    CMPLX_MUL_SC(a3x, a3y, (T)0, (T)1);                                       \
    FFT2_SC(a0x, a0y, a1x, a1y);                                               \
    FFT2_SC(a2x, a2y, a3x, a3y);                                               \
  } while (0)

#define IFFT8_SC(a0x, a0y, a1x, a1y, a2x, a2y, a3x, a3y,                      \
                 a4x, a4y, a5x, a5y, a6x, a6y, a7x, a7y)                      \
  do {                                                                         \
    FFT2_SC(a0x, a0y, a4x, a4y);                                               \
    FFT2_SC(a1x, a1y, a5x, a5y);                                               \
    FFT2_SC(a2x, a2y, a6x, a6y);                                               \
    FFT2_SC(a3x, a3y, a7x, a7y);                                               \
    CMPLX_MUL_SC(a5x, a5y, (T)1, (T)1);                                       \
    CM_FL_MUL_SC(a5x, a5y, (T)M_SQRT1_2);                                     \
    CMPLX_MUL_SC(a6x, a6y, (T)0, (T)1);                                       \
    CMPLX_MUL_SC(a7x, a7y, (T)-1, (T)1);                                      \
    CM_FL_MUL_SC(a7x, a7y, (T)M_SQRT1_2);                                     \
    IFFT4_SC(a0x, a0y, a1x, a1y, a2x, a2y, a3x, a3y);                         \
    IFFT4_SC(a4x, a4y, a5x, a5y, a6x, a6y, a7x, a7y);                         \
  } while (0)

void submit_fft_optimized(sycl::queue& q, T2* work, size_t globalsz,
                          size_t localsz) {
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<T, 1> smem(sycl::range<1>(8 * 8 * 9), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(globalsz), sycl::range<1>(localsz)),
        [=](sycl::nd_item<1> item) {
          int tid = item.get_local_id(0);
          int blockIdx = item.get_group(0) * 512 + tid;
          int hi = tid >> 3;
          int lo = tid & 7;
          T d0x, d0y, d1x, d1y, d2x, d2y, d3x, d3y;
          T d4x, d4y, d5x, d5y, d6x, d6y, d7x, d7y;

          FFT_LOAD_SC(d0x, d0y, work[blockIdx + 0 * 64]);
          FFT_LOAD_SC(d1x, d1y, work[blockIdx + 1 * 64]);
          FFT_LOAD_SC(d2x, d2y, work[blockIdx + 2 * 64]);
          FFT_LOAD_SC(d3x, d3y, work[blockIdx + 3 * 64]);
          FFT_LOAD_SC(d4x, d4y, work[blockIdx + 4 * 64]);
          FFT_LOAD_SC(d5x, d5y, work[blockIdx + 5 * 64]);
          FFT_LOAD_SC(d6x, d6y, work[blockIdx + 6 * 64]);
          FFT_LOAD_SC(d7x, d7y, work[blockIdx + 7 * 64]);

          FFT8_SC(d0x, d0y, d1x, d1y, d2x, d2y, d3x, d3y, d4x, d4y, d5x,
                  d5y, d6x, d6y, d7x, d7y);

          smem[hi * 8 + lo + 0 * 66] = d0x;
          CMPLX_MUL_EXP_SC(d4x, d4y, ((T)-2 * (T)M_PI * (T)1 / (T)512) * tid);
          smem[hi * 8 + lo + 1 * 66] = d4x;
          CMPLX_MUL_EXP_SC(d2x, d2y, ((T)-2 * (T)M_PI * (T)2 / (T)512) * tid);
          smem[hi * 8 + lo + 2 * 66] = d2x;
          CMPLX_MUL_EXP_SC(d6x, d6y, ((T)-2 * (T)M_PI * (T)3 / (T)512) * tid);
          smem[hi * 8 + lo + 3 * 66] = d6x;
          CMPLX_MUL_EXP_SC(d1x, d1y, ((T)-2 * (T)M_PI * (T)4 / (T)512) * tid);
          smem[hi * 8 + lo + 4 * 66] = d1x;
          CMPLX_MUL_EXP_SC(d5x, d5y, ((T)-2 * (T)M_PI * (T)5 / (T)512) * tid);
          smem[hi * 8 + lo + 5 * 66] = d5x;
          CMPLX_MUL_EXP_SC(d3x, d3y, ((T)-2 * (T)M_PI * (T)6 / (T)512) * tid);
          smem[hi * 8 + lo + 6 * 66] = d3x;
          CMPLX_MUL_EXP_SC(d7x, d7y, ((T)-2 * (T)M_PI * (T)7 / (T)512) * tid);
          smem[hi * 8 + lo + 7 * 66] = d7x;
          sycl::group_barrier(item.get_group());
          d0x = smem[lo * 66 + hi + 0 * 8];
          d1x = smem[lo * 66 + hi + 1 * 8];
          d2x = smem[lo * 66 + hi + 2 * 8];
          d3x = smem[lo * 66 + hi + 3 * 8];
          d4x = smem[lo * 66 + hi + 4 * 8];
          d5x = smem[lo * 66 + hi + 5 * 8];
          d6x = smem[lo * 66 + hi + 6 * 8];
          d7x = smem[lo * 66 + hi + 7 * 8];
          sycl::group_barrier(item.get_group());
          smem[hi * 8 + lo + 0 * 66] = d0y;
          smem[hi * 8 + lo + 1 * 66] = d4y;
          smem[hi * 8 + lo + 2 * 66] = d2y;
          smem[hi * 8 + lo + 3 * 66] = d6y;
          smem[hi * 8 + lo + 4 * 66] = d1y;
          smem[hi * 8 + lo + 5 * 66] = d5y;
          smem[hi * 8 + lo + 6 * 66] = d3y;
          smem[hi * 8 + lo + 7 * 66] = d7y;
          sycl::group_barrier(item.get_group());
          d0y = smem[lo * 66 + hi + 0 * 8];
          d1y = smem[lo * 66 + hi + 1 * 8];
          d2y = smem[lo * 66 + hi + 2 * 8];
          d3y = smem[lo * 66 + hi + 3 * 8];
          d4y = smem[lo * 66 + hi + 4 * 8];
          d5y = smem[lo * 66 + hi + 5 * 8];
          d6y = smem[lo * 66 + hi + 6 * 8];
          d7y = smem[lo * 66 + hi + 7 * 8];
          sycl::group_barrier(item.get_group());

          FFT8_SC(d0x, d0y, d1x, d1y, d2x, d2y, d3x, d3y, d4x, d4y, d5x,
                  d5y, d6x, d6y, d7x, d7y);

          smem[hi * 8 + lo + 0 * 72] = d0x;
          CMPLX_MUL_EXP_SC(d4x, d4y, ((T)-2 * (T)M_PI * (T)1 / (T)64) * hi);
          smem[hi * 8 + lo + 1 * 72] = d4x;
          CMPLX_MUL_EXP_SC(d2x, d2y, ((T)-2 * (T)M_PI * (T)2 / (T)64) * hi);
          smem[hi * 8 + lo + 2 * 72] = d2x;
          CMPLX_MUL_EXP_SC(d6x, d6y, ((T)-2 * (T)M_PI * (T)3 / (T)64) * hi);
          smem[hi * 8 + lo + 3 * 72] = d6x;
          CMPLX_MUL_EXP_SC(d1x, d1y, ((T)-2 * (T)M_PI * (T)4 / (T)64) * hi);
          smem[hi * 8 + lo + 4 * 72] = d1x;
          CMPLX_MUL_EXP_SC(d5x, d5y, ((T)-2 * (T)M_PI * (T)5 / (T)64) * hi);
          smem[hi * 8 + lo + 5 * 72] = d5x;
          CMPLX_MUL_EXP_SC(d3x, d3y, ((T)-2 * (T)M_PI * (T)6 / (T)64) * hi);
          smem[hi * 8 + lo + 6 * 72] = d3x;
          CMPLX_MUL_EXP_SC(d7x, d7y, ((T)-2 * (T)M_PI * (T)7 / (T)64) * hi);
          smem[hi * 8 + lo + 7 * 72] = d7x;
          sycl::group_barrier(item.get_group());
          d0x = smem[hi * 72 + lo + 0 * 8];
          d1x = smem[hi * 72 + lo + 1 * 8];
          d2x = smem[hi * 72 + lo + 2 * 8];
          d3x = smem[hi * 72 + lo + 3 * 8];
          d4x = smem[hi * 72 + lo + 4 * 8];
          d5x = smem[hi * 72 + lo + 5 * 8];
          d6x = smem[hi * 72 + lo + 6 * 8];
          d7x = smem[hi * 72 + lo + 7 * 8];
          sycl::group_barrier(item.get_group());
          smem[hi * 8 + lo + 0 * 72] = d0y;
          smem[hi * 8 + lo + 1 * 72] = d4y;
          smem[hi * 8 + lo + 2 * 72] = d2y;
          smem[hi * 8 + lo + 3 * 72] = d6y;
          smem[hi * 8 + lo + 4 * 72] = d1y;
          smem[hi * 8 + lo + 5 * 72] = d5y;
          smem[hi * 8 + lo + 6 * 72] = d3y;
          smem[hi * 8 + lo + 7 * 72] = d7y;
          sycl::group_barrier(item.get_group());
          d0y = smem[hi * 72 + lo + 0 * 8];
          d1y = smem[hi * 72 + lo + 1 * 8];
          d2y = smem[hi * 72 + lo + 2 * 8];
          d3y = smem[hi * 72 + lo + 3 * 8];
          d4y = smem[hi * 72 + lo + 4 * 8];
          d5y = smem[hi * 72 + lo + 5 * 8];
          d6y = smem[hi * 72 + lo + 6 * 8];
          d7y = smem[hi * 72 + lo + 7 * 8];

          FFT8_SC(d0x, d0y, d1x, d1y, d2x, d2y, d3x, d3y, d4x, d4y, d5x,
                  d5y, d6x, d6y, d7x, d7y);

          FFT_STORE_SC(work[blockIdx + 0 * 64], d0x, d0y);
          FFT_STORE_SC(work[blockIdx + 1 * 64], d4x, d4y);
          FFT_STORE_SC(work[blockIdx + 2 * 64], d2x, d2y);
          FFT_STORE_SC(work[blockIdx + 3 * 64], d6x, d6y);
          FFT_STORE_SC(work[blockIdx + 4 * 64], d1x, d1y);
          FFT_STORE_SC(work[blockIdx + 5 * 64], d5x, d5y);
          FFT_STORE_SC(work[blockIdx + 6 * 64], d3x, d3y);
          FFT_STORE_SC(work[blockIdx + 7 * 64], d7x, d7y);
        });
  });
}

void submit_ifft_optimized(sycl::queue& q, T2* work, size_t globalsz,
                           size_t localsz) {
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<T, 1> smem(sycl::range<1>(8 * 8 * 9), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(globalsz), sycl::range<1>(localsz)),
        [=](sycl::nd_item<1> item) {
          int tid = item.get_local_id(0);
          int blockIdx = item.get_group(0) * 512 + tid;
          int hi = tid >> 3;
          int lo = tid & 7;
          T d0x, d0y, d1x, d1y, d2x, d2y, d3x, d3y;
          T d4x, d4y, d5x, d5y, d6x, d6y, d7x, d7y;

          FFT_LOAD_SC(d0x, d0y, work[blockIdx + 0 * 64]);
          FFT_LOAD_SC(d1x, d1y, work[blockIdx + 1 * 64]);
          FFT_LOAD_SC(d2x, d2y, work[blockIdx + 2 * 64]);
          FFT_LOAD_SC(d3x, d3y, work[blockIdx + 3 * 64]);
          FFT_LOAD_SC(d4x, d4y, work[blockIdx + 4 * 64]);
          FFT_LOAD_SC(d5x, d5y, work[blockIdx + 5 * 64]);
          FFT_LOAD_SC(d6x, d6y, work[blockIdx + 6 * 64]);
          FFT_LOAD_SC(d7x, d7y, work[blockIdx + 7 * 64]);

          IFFT8_SC(d0x, d0y, d1x, d1y, d2x, d2y, d3x, d3y, d4x, d4y,
                   d5x, d5y, d6x, d6y, d7x, d7y);

          smem[hi * 8 + lo + 0 * 66] = d0x;
          CMPLX_MUL_EXP_SC(d4x, d4y, ((T)2 * (T)M_PI * (T)1 / (T)512) * tid);
          smem[hi * 8 + lo + 1 * 66] = d4x;
          CMPLX_MUL_EXP_SC(d2x, d2y, ((T)2 * (T)M_PI * (T)2 / (T)512) * tid);
          smem[hi * 8 + lo + 2 * 66] = d2x;
          CMPLX_MUL_EXP_SC(d6x, d6y, ((T)2 * (T)M_PI * (T)3 / (T)512) * tid);
          smem[hi * 8 + lo + 3 * 66] = d6x;
          CMPLX_MUL_EXP_SC(d1x, d1y, ((T)2 * (T)M_PI * (T)4 / (T)512) * tid);
          smem[hi * 8 + lo + 4 * 66] = d1x;
          CMPLX_MUL_EXP_SC(d5x, d5y, ((T)2 * (T)M_PI * (T)5 / (T)512) * tid);
          smem[hi * 8 + lo + 5 * 66] = d5x;
          CMPLX_MUL_EXP_SC(d3x, d3y, ((T)2 * (T)M_PI * (T)6 / (T)512) * tid);
          smem[hi * 8 + lo + 6 * 66] = d3x;
          CMPLX_MUL_EXP_SC(d7x, d7y, ((T)2 * (T)M_PI * (T)7 / (T)512) * tid);
          smem[hi * 8 + lo + 7 * 66] = d7x;
          sycl::group_barrier(item.get_group());
          d0x = smem[lo * 66 + hi + 0 * 8];
          d1x = smem[lo * 66 + hi + 1 * 8];
          d2x = smem[lo * 66 + hi + 2 * 8];
          d3x = smem[lo * 66 + hi + 3 * 8];
          d4x = smem[lo * 66 + hi + 4 * 8];
          d5x = smem[lo * 66 + hi + 5 * 8];
          d6x = smem[lo * 66 + hi + 6 * 8];
          d7x = smem[lo * 66 + hi + 7 * 8];
          sycl::group_barrier(item.get_group());
          smem[hi * 8 + lo + 0 * 66] = d0y;
          smem[hi * 8 + lo + 1 * 66] = d4y;
          smem[hi * 8 + lo + 2 * 66] = d2y;
          smem[hi * 8 + lo + 3 * 66] = d6y;
          smem[hi * 8 + lo + 4 * 66] = d1y;
          smem[hi * 8 + lo + 5 * 66] = d5y;
          smem[hi * 8 + lo + 6 * 66] = d3y;
          smem[hi * 8 + lo + 7 * 66] = d7y;
          sycl::group_barrier(item.get_group());
          d0y = smem[lo * 66 + hi + 0 * 8];
          d1y = smem[lo * 66 + hi + 1 * 8];
          d2y = smem[lo * 66 + hi + 2 * 8];
          d3y = smem[lo * 66 + hi + 3 * 8];
          d4y = smem[lo * 66 + hi + 4 * 8];
          d5y = smem[lo * 66 + hi + 5 * 8];
          d6y = smem[lo * 66 + hi + 6 * 8];
          d7y = smem[lo * 66 + hi + 7 * 8];
          sycl::group_barrier(item.get_group());

          IFFT8_SC(d0x, d0y, d1x, d1y, d2x, d2y, d3x, d3y, d4x, d4y,
                   d5x, d5y, d6x, d6y, d7x, d7y);

          smem[hi * 8 + lo + 0 * 72] = d0x;
          CMPLX_MUL_EXP_SC(d4x, d4y, ((T)2 * (T)M_PI * (T)1 / (T)64) * hi);
          smem[hi * 8 + lo + 1 * 72] = d4x;
          CMPLX_MUL_EXP_SC(d2x, d2y, ((T)2 * (T)M_PI * (T)2 / (T)64) * hi);
          smem[hi * 8 + lo + 2 * 72] = d2x;
          CMPLX_MUL_EXP_SC(d6x, d6y, ((T)2 * (T)M_PI * (T)3 / (T)64) * hi);
          smem[hi * 8 + lo + 3 * 72] = d6x;
          CMPLX_MUL_EXP_SC(d1x, d1y, ((T)2 * (T)M_PI * (T)4 / (T)64) * hi);
          smem[hi * 8 + lo + 4 * 72] = d1x;
          CMPLX_MUL_EXP_SC(d5x, d5y, ((T)2 * (T)M_PI * (T)5 / (T)64) * hi);
          smem[hi * 8 + lo + 5 * 72] = d5x;
          CMPLX_MUL_EXP_SC(d3x, d3y, ((T)2 * (T)M_PI * (T)6 / (T)64) * hi);
          smem[hi * 8 + lo + 6 * 72] = d3x;
          CMPLX_MUL_EXP_SC(d7x, d7y, ((T)2 * (T)M_PI * (T)7 / (T)64) * hi);
          smem[hi * 8 + lo + 7 * 72] = d7x;
          sycl::group_barrier(item.get_group());
          d0x = smem[hi * 72 + lo + 0 * 8];
          d1x = smem[hi * 72 + lo + 1 * 8];
          d2x = smem[hi * 72 + lo + 2 * 8];
          d3x = smem[hi * 72 + lo + 3 * 8];
          d4x = smem[hi * 72 + lo + 4 * 8];
          d5x = smem[hi * 72 + lo + 5 * 8];
          d6x = smem[hi * 72 + lo + 6 * 8];
          d7x = smem[hi * 72 + lo + 7 * 8];
          sycl::group_barrier(item.get_group());
          smem[hi * 8 + lo + 0 * 72] = d0y;
          smem[hi * 8 + lo + 1 * 72] = d4y;
          smem[hi * 8 + lo + 2 * 72] = d2y;
          smem[hi * 8 + lo + 3 * 72] = d6y;
          smem[hi * 8 + lo + 4 * 72] = d1y;
          smem[hi * 8 + lo + 5 * 72] = d5y;
          smem[hi * 8 + lo + 6 * 72] = d3y;
          smem[hi * 8 + lo + 7 * 72] = d7y;
          sycl::group_barrier(item.get_group());
          d0y = smem[hi * 72 + lo + 0 * 8];
          d1y = smem[hi * 72 + lo + 1 * 8];
          d2y = smem[hi * 72 + lo + 2 * 8];
          d3y = smem[hi * 72 + lo + 3 * 8];
          d4y = smem[hi * 72 + lo + 4 * 8];
          d5y = smem[hi * 72 + lo + 5 * 8];
          d6y = smem[hi * 72 + lo + 6 * 8];
          d7y = smem[hi * 72 + lo + 7 * 8];

          IFFT8_SC(d0x, d0y, d1x, d1y, d2x, d2y, d3x, d3y, d4x, d4y,
                   d5x, d5y, d6x, d6y, d7x, d7y);

          d0x = d0x / (T)512;
          d0y = d0y / (T)512;
          d1x = d1x / (T)512;
          d1y = d1y / (T)512;
          d2x = d2x / (T)512;
          d2y = d2y / (T)512;
          d3x = d3x / (T)512;
          d3y = d3y / (T)512;
          d4x = d4x / (T)512;
          d4y = d4y / (T)512;
          d5x = d5x / (T)512;
          d5y = d5y / (T)512;
          d6x = d6x / (T)512;
          d6y = d6y / (T)512;
          d7x = d7x / (T)512;
          d7y = d7y / (T)512;

          FFT_STORE_SC(work[blockIdx + 0 * 64], d0x, d0y);
          FFT_STORE_SC(work[blockIdx + 1 * 64], d4x, d4y);
          FFT_STORE_SC(work[blockIdx + 2 * 64], d2x, d2y);
          FFT_STORE_SC(work[blockIdx + 3 * 64], d6x, d6y);
          FFT_STORE_SC(work[blockIdx + 4 * 64], d1x, d1y);
          FFT_STORE_SC(work[blockIdx + 5 * 64], d5x, d5y);
          FFT_STORE_SC(work[blockIdx + 6 * 64], d3x, d3y);
          FFT_STORE_SC(work[blockIdx + 7 * 64], d7x, d7y);
        });
  });
}
