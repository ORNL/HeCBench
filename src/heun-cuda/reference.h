#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

template <typename Fn>
void parallel_for(int n, Fn&& fn) {
  if (n <= 0) return;
  unsigned threads = std::thread::hardware_concurrency();
  if (threads == 0) threads = 1;
  if (threads > static_cast<unsigned>(n)) threads = static_cast<unsigned>(n);
  if (threads == 1) {
    for (int i = 0; i < n; ++i) fn(i);
    return;
  }

  std::vector<std::thread> pool;
  pool.reserve(threads);
  const int chunk = (n + static_cast<int>(threads) - 1) / static_cast<int>(threads);
  for (unsigned t = 0; t < threads; ++t) {
    const int begin = static_cast<int>(t) * chunk;
    const int end = std::min(n, begin + chunk);
    pool.emplace_back([&, begin, end] {
      for (int i = begin; i < end; ++i) fn(i);
    });
  }
  for (std::thread& th : pool) th.join();
}

// NCHW cross-correlation, 3x3, pad 1, stride 1. Each task owns one output row.
void conv3x3_nchw(const float* in, const float* weight, float* out, int batch,
                  int channels, int height, int width) {
  const size_t spatial = static_cast<size_t>(height) * static_cast<size_t>(width);
  const int tasks = batch * channels * height;
  parallel_for(tasks, [&](int task) {
    const int plane = channels * height;
    const int n = task / plane;
    const int rem = task - n * plane;
    const int oc = rem / height;
    const int oh = rem - oc * height;

    float* y = out + (static_cast<size_t>(n) * channels + oc) * spatial +
               static_cast<size_t>(oh) * width;
    for (int ow = 0; ow < width; ++ow) y[ow] = 0.f;

    for (int ic = 0; ic < channels; ++ic) {
      const float* x = in + (static_cast<size_t>(n) * channels + ic) * spatial;
      const float* k = weight + (static_cast<size_t>(oc) * channels + ic) * 9;
      const bool interior =
          height >= 3 && width >= 2 && oh > 0 && oh + 1 < height;
      if (interior) {
        const float* r0 = x + static_cast<size_t>(oh - 1) * width;
        const float* r1 = x + static_cast<size_t>(oh) * width;
        const float* r2 = x + static_cast<size_t>(oh + 1) * width;
        const float k00 = k[0], k01 = k[1], k02 = k[2];
        const float k10 = k[3], k11 = k[4], k12 = k[5];
        const float k20 = k[6], k21 = k[7], k22 = k[8];
        y[0] += r0[0] * k01 + r0[1] * k02 + r1[0] * k11 + r1[1] * k12 +
                r2[0] * k21 + r2[1] * k22;
        for (int ow = 1; ow < width - 1; ++ow) {
          y[ow] += r0[ow - 1] * k00 + r0[ow] * k01 + r0[ow + 1] * k02 +
                   r1[ow - 1] * k10 + r1[ow] * k11 + r1[ow + 1] * k12 +
                   r2[ow - 1] * k20 + r2[ow] * k21 + r2[ow + 1] * k22;
        }
        const int ow = width - 1;
        y[ow] += r0[ow - 1] * k00 + r0[ow] * k01 + r1[ow - 1] * k10 +
                 r1[ow] * k11 + r2[ow - 1] * k20 + r2[ow] * k21;
      } else {
        for (int ow = 0; ow < width; ++ow) {
          float acc = y[ow];
          for (int kh = 0; kh < 3; ++kh) {
            const int ih = oh + kh - 1;
            if (static_cast<unsigned>(ih) >= static_cast<unsigned>(height))
              continue;
            const float* row = x + static_cast<size_t>(ih) * width;
            const float* kk = k + kh * 3;
            for (int kw = 0; kw < 3; ++kw) {
              const int iw = ow + kw - 1;
              if (static_cast<unsigned>(iw) >= static_cast<unsigned>(width))
                continue;
              acc += row[iw] * kk[kw];
            }
          }
          y[ow] = acc;
        }
      }
    }
  });
}

void silu_ref(float* x, size_t n, float sigma) {
  const float denom = 1.f + sigma * sigma;
  auto apply = [&](size_t i) {
    const float v = x[i];
    x[i] = (v / (1.f + expf(-v))) / denom;
  };
  if (n > static_cast<size_t>(0x7fffffff)) {
    for (size_t i = 0; i < n; ++i) apply(i);
    return;
  }
  parallel_for(static_cast<int>(n), [&](int i) { apply(static_cast<size_t>(i)); });
}

void edm_fill_weights(float* x, size_t n, unsigned seed) {
  for (size_t i = 0; i < n; ++i) {
    unsigned v = static_cast<unsigned>(i) * 747796405u + seed;
    v ^= v >> 16;
    x[i] = (static_cast<float>(v & 0xffffu) * (1.f / 32768.f) - 1.f) * 0.02f;
  }
}

void edm_heun_reference(const float* x, const float* w1, const float* w2,
                        float* out, const float* times, int steps, int batch,
                        int channels, int height, int width) {
  const size_t elements = static_cast<size_t>(batch) * channels * height * width;
  std::vector<float> current(x, x + elements);
  std::vector<float> next(elements);
  std::vector<float> hidden(elements);
  std::vector<float> x0(elements);
  std::vector<float> x0_cur(elements);
  std::vector<float> prime(elements);

  for (int step = 0; step < steps; ++step) {
    const float t_cur = times[step];
    const float t_next = times[step + 1];
    const float h = t_next - t_cur;
    const bool correct = t_next != 0.f;

    conv3x3_nchw(current.data(), w1, hidden.data(), batch, channels, height,
                 width);
    silu_ref(hidden.data(), elements, t_cur);
    conv3x3_nchw(hidden.data(), w2, x0.data(), batch, channels, height, width);

    if (correct) {
      parallel_for(static_cast<int>(elements), [&](int idx) {
        const size_t i = static_cast<size_t>(idx);
        const float d_cur = (current[i] - x0[i]) / t_cur;
        prime[i] = current[i] + h * d_cur;
      });
      x0_cur.swap(x0);
      conv3x3_nchw(prime.data(), w1, hidden.data(), batch, channels, height,
                   width);
      silu_ref(hidden.data(), elements, t_next);
      conv3x3_nchw(hidden.data(), w2, x0.data(), batch, channels, height, width);
      parallel_for(static_cast<int>(elements), [&](int idx) {
        const size_t i = static_cast<size_t>(idx);
        const float d_cur = (current[i] - x0_cur[i]) / t_cur;
        const float d_prime = (current[i] + h * d_cur - x0[i]) / t_next;
        next[i] = current[i] + 0.5f * h * (d_cur + d_prime);
      });
    } else {
      parallel_for(static_cast<int>(elements), [&](int idx) {
        const size_t i = static_cast<size_t>(idx);
        const float d_cur = (current[i] - x0[i]) / t_cur;
        next[i] = current[i] + h * d_cur;
      });
    }
    current.swap(next);
  }
  std::copy(current.begin(), current.end(), out);
}
