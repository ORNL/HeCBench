#pragma once
//
// Configuration and host reference for the ragged
// (variable-valid-length) top-k configurations.
//

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

// sweep

inline constexpr int32_t RAGGED_HIDDEN[] = {4096, 32768};
inline constexpr int32_t RAGGED_NEXT_N[] = {1, 4};
inline constexpr int32_t RAGGED_TOPKS[]  = {64, 2048};

// invokeComputeTopkLastDimWorkspaceSize() in topk_per_row_kernels.h sizes the
// scratch buffer with a hardcoded k = 2048 internally. A larger k here would
// silently under-allocate it -- a heap overflow with no diagnostic, on the
// device. Catch it at compile time instead.
inline constexpr int32_t RAGGED_WORKSPACE_ASSUMED_K = 2048;
constexpr bool ragged_topks_fit_workspace()
{
  for (int32_t k : RAGGED_TOPKS)
    if (k > RAGGED_WORKSPACE_ASSUMED_K) return false;
  return true;
}
static_assert(ragged_topks_fit_workspace(),
              "RAGGED_TOPKS exceeds the k that the workspace sizer assumes; "
              "raise it there before raising it here");

// A decode batch is bounded by max_num_seqs, typically <= 256. rows =
// requests * next_n, so binding the request count straight to argv's
// batch_size would let the ragged sweep allocate up to next_n times the
// fixed-extent sweep's footprint -- 1.5 GiB for the input alone at the
// canonical `make run` arguments, which the smaller GPUs the CUDA and SYCL
// ports target may not have. Cap it, and say so when the cap bites.
inline constexpr int32_t RAGGED_MAX_REQUESTS = 256;

inline int32_t ragged_request_count(int32_t batch_size)
{
  return std::max(1, std::min(batch_size, RAGGED_MAX_REQUESTS));
}

// Poison the output index buffer with this before each launch. It must NOT be
// -1: pre-filling with the sentinel under test would make "slot holds -1"
// ambiguous between "the kernel wrote the sentinel" and "the kernel never
// touched this slot", which is the whole question.
inline constexpr int RAGGED_POISON_BYTE = 0xAB;

// Derived, never written out by hand: memset fills every byte, so the 32-bit
// value is the byte smeared four times. Spelling both independently would let
// them drift, and a stale sentinel would misreport skipped slots as "real
// index" -- the exact opposite of the truth.
inline constexpr int32_t RAGGED_UNTOUCHED =
    (int32_t)(0x01010101u * (uint32_t)(RAGGED_POISON_BYTE & 0xFF));

static_assert(RAGGED_UNTOUCHED != -1,
              "RAGGED_POISON_BYTE must not be 0xFF: it would smear to the -1 "
              "sentinel and make 'kernel wrote -1' indistinguishable from "
              "'untouched'");

// the shapes

enum class RaggedDist { Uniform, Bimodal, Skewed, Random };

inline constexpr RaggedDist RAGGED_DISTS[] = {
    RaggedDist::Uniform, RaggedDist::Bimodal, RaggedDist::Skewed,
    RaggedDist::Random};

inline const char* ragged_dist_name(RaggedDist d)
{
  switch (d) {
    case RaggedDist::Uniform: return "uniform";
    case RaggedDist::Bimodal: return "bimodal";
    case RaggedDist::Skewed:  return "skewed";
    default:                  return "random";
  }
}

// One configuration, fully described. seq_lens (per request, what the device
// receives) and row_lens (per row, what the kernel derives and the host
// reference checks against) are built together so they cannot be passed as a
// mismatched pair.
struct RaggedCase {
  RaggedDist dist;
  int32_t max_len;
  int32_t next_n;
  int32_t rows;
  std::vector<int32_t> seq_lens;  // [num_req]
  std::vector<int32_t> row_lens;  // [rows]
};

// Builds both arrays for one (dist, next_n, max_len).
//
//   uniform - every request at full width. The control: this is the
//             fixed-extent sweep reached through the ragged code path, so the
//             difference is the cost of the ragged machinery itself.
//   bimodal - half at full width, half at the floor. Worst case for a static
//             row->block assignment.
//   skewed  - roughly a real decode batch: ~90% short, a few very long.
//   random  - uniform random over the whole range.
//
// The floor keeps every row's staircase entry positive.
inline RaggedCase ragged_make_case(RaggedDist d, int32_t num_req,
                                   int32_t max_len, int32_t next_n,
                                   uint32_t seed = 1234u)
{
  const int32_t lo = std::max(next_n, 8);

  RaggedCase c;
  c.dist = d;
  c.max_len = max_len;
  c.next_n = next_n;
  c.rows = num_req * next_n;
  c.seq_lens.resize(num_req);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::uniform_int_distribution<int32_t> uni(lo, max_len);
  for (int32_t b = 0; b < num_req; b++) {
    int32_t v;
    switch (d) {
      case RaggedDist::Uniform: v = max_len; break;
      case RaggedDist::Bimodal: v = (b % 2) ? max_len : lo; break;
      case RaggedDist::Skewed:
        v = (u01(rng) < 0.9) ? lo + (int32_t)(u01(rng) * (max_len / 16))
                             : max_len / 2 + (int32_t)(u01(rng) * (max_len / 2));
        break;
      default: v = uni(rng); break;
    }
    c.seq_lens[b] = std::min(std::max(v, lo), max_len);
  }

  // The staircase the kernel computes internally
  // (topk_per_row_kernels.h, Phase::Decode):
  //
  //     row_len = seq_lens[row / next_n] - next_n + (row % next_n) + 1
  //
  // Mirrored here on the host so every backend uploads the same seq_lens and
  // checks against the same expected lengths, rather than each reimplementing
  // the formula in a device kernel.
  c.row_lens.resize(c.rows);
  for (int32_t i = 0; i < c.rows; i++) {
    const int32_t v = c.seq_lens[i / next_n] - next_n + (i % next_n) + 1;
    c.row_lens[i] = std::min(std::max(v, 0), max_len);
  }
  return c;
}

// host reference

struct RaggedCheck {
  bool ok = false;
  int32_t first_bad = -1;
  int64_t pad_slots = 0;      // output slots that cannot be filled (k > row_len)
  int64_t pad_neg1 = 0;       // ... holding -1, i.e. the kernel wrote a sentinel
  int64_t pad_untouched = 0;  // ... still poison, i.e. the kernel skipped them
  int64_t pad_other = 0;      // ... holding something else (a real index)
  int32_t pad_sample = 0;     // one example of "something else"
};

// Top-k of each row's OWN prefix. Values are compared as a sorted set, which
// is unambiguous because each row holds a shuffled permutation (no ties).
// Indices are bounds-checked against the row's own length, which is the
// property the ragged path can actually get wrong.
inline RaggedCheck ragged_verify(const std::vector<float>& h_x,
                                 const std::vector<float>& h_val,
                                 const std::vector<int32_t>& h_ids,
                                 const RaggedCase& c, int32_t k)
{
  RaggedCheck chk;
  std::vector<float> prefix, got;

  for (int32_t r = 0; r < c.rows; r++) {
    const int32_t len = c.max_len;
    const int32_t L = c.row_lens[r];  // ragged_make_case clamps this to max_len
    const int32_t nv = std::min(k, L);

    prefix.assign(h_x.begin() + (size_t)r * len,
                  h_x.begin() + (size_t)r * len + L);
    std::partial_sort(prefix.begin(), prefix.begin() + nv, prefix.end(),
                      std::greater<float>());

    got.assign(h_val.begin() + (size_t)r * k, h_val.begin() + (size_t)r * k + nv);
    std::sort(got.begin(), got.end(), std::greater<float>());

    if (!std::equal(prefix.begin(), prefix.begin() + nv, got.begin())) {
      chk.first_bad = r;
      return chk;
    }
    for (int32_t i = 0; i < nv; i++) {
      const int32_t id = h_ids[(size_t)r * k + i];
      if (id < 0 || id >= L) { chk.first_bad = r; return chk; }
    }
    for (int32_t i = nv; i < k; i++) {
      const int32_t id = h_ids[(size_t)r * k + i];
      chk.pad_slots++;
      if (id == -1) chk.pad_neg1++;
      else if (id == RAGGED_UNTOUCHED) chk.pad_untouched++;
      else { if (!chk.pad_other) chk.pad_sample = id; chk.pad_other++; }
    }
  }
  chk.ok = true;
  return chk;
}

// reporting

inline void ragged_print_banner(int32_t batch_size, int32_t num_req)
{
  printf("\n\n================ ragged (variable valid length) ================\n");
  if (num_req != batch_size)
    printf("requests capped at %d (argv gave %d); rows = requests * next_n\n",
           num_req, batch_size);
  printf("Each row selects only over its own prefix; the kernel derives that "
         "prefix from\nseq_lens and next_n. `uniform` is the control: "
         "full-width rows reached through\nthe ragged path, so it is the sweep "
         "above with the ragged machinery engaged.\n");
}

inline void ragged_print_config(const RaggedCase& c, int32_t k)
{
  double mean = 0.0;
  int32_t mx = 0, below = 0;
  for (int32_t v : c.row_lens) {
    mean += v;
    mx = std::max(mx, v);
    if (v < k) below++;
  }
  mean /= (double)c.row_lens.size();

  printf("\nhidden size: %d, next_n: %d, rows: %d, dist: %s, topk: %d\n",
         c.max_len, c.next_n, c.rows, ragged_dist_name(c.dist), k);
  printf("  imbalance(max/mean)=%.2f  rows with row_len<k: %d/%d\n",
         mx / mean, below, c.rows);
}

inline void ragged_print_result(double us, const RaggedCheck& c)
{
  printf("  average execution time of ragged topk: %9.2f (us)  %s",
         us, c.ok ? "PASS" : "FAIL");
  if (!c.ok) printf(" (first bad row %d)", c.first_bad);
  if (c.pad_slots) {
    printf("  [unfillable k>row_len: %lld -> sentinel(-1): %lld, untouched: %lld,"
           " real idx: %lld",
           (long long)c.pad_slots, (long long)c.pad_neg1,
           (long long)c.pad_untouched, (long long)c.pad_other);
    if (c.pad_other) printf(" e.g. %d", c.pad_sample);
    printf("]");
  }
  printf("\n");
}
