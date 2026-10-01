#ifndef FLASH_MLA_SOFTMAX_H
#define FLASH_MLA_SOFTMAX_H

#include <sycl/sycl.hpp>
#include "reference.h"   // DIM, D_V, PAGE, NEG_LARGE

typedef sycl::ext::oneapi::bfloat16 bf16_t;

// One work-group per output row (b*H + h). Reads scaled scores S[row, 0:n],
// writes the row-normalized probabilities P (bf16) and the log-sum-exp (fp32).
// A grid-stride loop over rows lets `rows` exceed the number of work-groups.
static inline void mla_softmax(sycl::queue &q, const float *S, bf16_t *P,
                               float *lse, int n, size_t rows) {
  const size_t max_groups = 65535;
  const size_t groups = std::min(rows, max_groups);
  const int WG = 256;
  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) {
          auto g = it.get_group();
          const int tid = it.get_local_id(0);
          const size_t ng = it.get_group_range(0);
          for (size_t row = it.get_group(0); row < rows; row += ng) {
            const float *s = S + row * n;
            bf16_t      *p = P + row * n;

            float m = NEG_LARGE;
            for (int j = tid; j < n; j += WG) m = sycl::fmax(m, s[j]);
            m = sycl::reduce_over_group(g, m, sycl::maximum<float>());

            float sum = 0.f;
            for (int j = tid; j < n; j += WG) sum += sycl::exp(s[j] - m);
            sum = sycl::reduce_over_group(g, sum, sycl::plus<float>());

            const float inv = (sum > 0.f) ? 1.f / sum : 0.f;
            for (int j = tid; j < n; j += WG)
              p[j] = bf16_t(sycl::exp(s[j] - m) * inv);

            if (tid == 0)
              lse[row] = (sum > 0.f) ? (m + sycl::log(sum)) : NEG_LARGE;
          }
        });
  });
}

#endif  // FLASH_MLA_SOFTMAX_H
