#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

// Zipf/power-law group sizes with empty groups. Offsets and values are signed 64-bit 
using offset_t = std::int64_t;
using value_t = std::int64_t;

static constexpr double kZipfS = 1.2;
static constexpr unsigned kSeed = 123u;

struct SegProblem {
  offset_t num_items;
  offset_t num_segments;
  offset_t num_empty;
  offset_t min_nz;
  offset_t max_nz;
  double avg_nz;
  std::vector<offset_t> offsets; // num_segments + 1
  std::vector<value_t> values;
};

inline SegProblem make_zipf_problem(offset_t num_items)
{
  SegProblem p;
  p.num_items = num_items;

  const offset_t n_nonzero =
      std::min(num_items, std::max(offset_t(256), num_items / 64));
  p.num_empty = std::max(offset_t(1), n_nonzero / 20);
  p.num_segments = n_nonzero + p.num_empty;

  std::vector<double> w(n_nonzero);
  double sumw = 0.0;
  for (offset_t i = 0; i < n_nonzero; i++) {
    w[i] = 1.0 / std::pow(static_cast<double>(i + 1), kZipfS);
    sumw += w[i];
  }

  std::vector<offset_t> lengths(p.num_segments, 0);
  offset_t assigned = 0;
  for (offset_t i = 0; i < n_nonzero; i++) {
    offset_t take =
        std::max(offset_t(1),
                 static_cast<offset_t>(w[i] / sumw * static_cast<double>(num_items)));
    lengths[i] = take;
    assigned += take;
  }
  if (assigned < num_items) {
    lengths[0] += num_items - assigned;
  } else if (assigned > num_items) {
    offset_t extra = assigned - num_items;
    for (offset_t i = 0; extra > 0 && i < n_nonzero; i++) {
      const offset_t take = std::min(lengths[i] - 1, extra);
      lengths[i] -= take;
      extra -= take;
    }
  }

  std::mt19937 rng(kSeed);
  std::shuffle(lengths.begin(), lengths.end(), rng);

  p.offsets.resize(p.num_segments + 1);
  p.offsets[0] = 0;
  p.min_nz = std::numeric_limits<offset_t>::max();
  p.max_nz = 0;
  offset_t nz_sum = 0;
  offset_t nz_count = 0;
  for (offset_t i = 0; i < p.num_segments; i++) {
    p.offsets[i + 1] = p.offsets[i] + lengths[i];
    if (lengths[i] > 0) {
      p.min_nz = std::min(p.min_nz, lengths[i]);
      p.max_nz = std::max(p.max_nz, lengths[i]);
      nz_sum += lengths[i];
      nz_count++;
    }
  }
  p.avg_nz = nz_count ? static_cast<double>(nz_sum) / nz_count : 0.0;

  std::uniform_int_distribution<int> dist(1, 16);
  p.values.resize(num_items);
  for (offset_t i = 0; i < num_items; i++)
    p.values[i] = dist(rng);

  return p;
}

inline void print_problem(const SegProblem &p)
{
  printf("num_items = %lld\n", static_cast<long long>(p.num_items));
  printf("num_segments = %lld (empty = %lld, zipf_s = %.1f)\n",
         static_cast<long long>(p.num_segments),
         static_cast<long long>(p.num_empty), kZipfS);
  printf("nonzero group size min/avg/max = %lld / %.1f / %lld\n",
         static_cast<long long>(p.min_nz), p.avg_nz,
         static_cast<long long>(p.max_nz));
}

inline void cpu_segmented(const SegProblem &p, std::vector<value_t> &sum,
                          std::vector<value_t> &mn, std::vector<value_t> &mx)
{
  const offset_t n = p.num_segments;
  sum.assign(n, 0);
  mn.assign(n, std::numeric_limits<value_t>::max());
  mx.assign(n, std::numeric_limits<value_t>::lowest());
  for (offset_t s = 0; s < n; s++) {
    const offset_t begin = p.offsets[s];
    const offset_t end = p.offsets[s + 1];
    if (begin == end)
      continue;
    value_t ssum = 0;
    value_t smin = std::numeric_limits<value_t>::max();
    value_t smax = std::numeric_limits<value_t>::lowest();
    for (offset_t i = begin; i < end; i++) {
      const value_t v = p.values[i];
      ssum += v;
      smin = std::min(smin, v);
      smax = std::max(smax, v);
    }
    sum[s] = ssum;
    mn[s] = smin;
    mx[s] = smax;
  }
}

inline int count_errors(const std::vector<value_t> &got,
                        const std::vector<value_t> &ref, const char *op)
{
  int errors = 0;
  const offset_t n = ref.size();
  for (offset_t i = 0; i < n; i++) {
    if (got[i] != ref[i]) {
      errors++;
      if (errors < 10)
        printf("%s segment %lld got %lld expected %lld\n", op,
               static_cast<long long>(i), static_cast<long long>(got[i]),
               static_cast<long long>(ref[i]));
    }
  }
  if (errors > 0)
    printf("%s: segmented reduction does not agree with the reference! %d "
           "errors!\n",
           op, errors);
  printf("%s\n", errors == 0 ? "PASS" : "FAIL");
  return errors;
}
