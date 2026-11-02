// Host routing and reference for the intra-node MoE combine.
// Each token draws `topk` distinct experts from the full pool
// (experts per rank * world). A fixed hash of (source rank, token) stands in
// for the random scores of the balanced-gate baseline (random scores, then
// top-k), so every rank builds the same route and the host reference matches
// the device.
// Expert e lives on rank e / experts, the usual contiguous placement.
// Measured on this hash at 8 ranks and 4096 tokens, a token's experts land on
// 5.29 ranks on average. In that draw every token's experts occupy more than
// one rank.
//
// Dispatch still sends the token once per destination rank, as MORI does. A
// later expert whose rank was already chosen for this token is dropped and is
// not read. Each destination packs the kept arrivals in source-rank, token,
// then top-k order. MORI adds those kept hidden vectors and reduces the router
// weights in a separate buffer. This benchmark scales each kept vector by the
// router weight of the expert that kept the slot:
//   out[token] = sum_{kept k} weight[token, k] * expert_output[dest_rank, slot]
//
// Bandwidth follows the MORI EP benchmark. Algo bytes are
// recv_tokens * hidden * sizeof(bf16): one hidden vector per kept token,
// including tokens that stay on this rank. Fabric bytes count only reads from
// other ranks. The driver reports the slowest rank and uses that rank's
// counts, so the numerator and the time belong to the same rank.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define COMBINE_HD __host__ __device__
#else
#define COMBINE_HD
#endif

struct CombineRoute {
  int world = 1;
  int tokens = 0;
  int topk = 1;
  int hidden = 1;
  int experts = 1;
  // dest/slot are indexed [source_rank, token, k].
  std::vector<int> dest;
  std::vector<int> slot;
  std::vector<int> recv;

  // Flat index of expert slot (source rank, token, k).
  std::size_t index(int src, int token, int k) const {
    return (static_cast<std::size_t>(src) * tokens + token) * topk + k;
  }
};

// Deterministic expert value in [0, 251).
COMBINE_HD inline int stage_mod(int rank, int slot, int h) {
  const std::int64_t m = static_cast<std::int64_t>(rank + 1) * 17 +
                         static_cast<std::int64_t>(slot) * 3 + (h % 251);
  return static_cast<int>(m % 251);
}

// stage_mod scaled into [0, 1).
COMBINE_HD inline float stage_unit(int rank, int slot, int h) {
  return static_cast<float>(stage_mod(rank, slot, h)) / 251.0f;
}

// Router weight (1 + (token * topk + k) % 7) / 8.
COMBINE_HD inline float weight_of(int token, int k, int topk) {
  return static_cast<float>((token * topk + k) % 7 + 1) / 8.0f;
}

// Round-to-nearest-even bf16, matching the CUDA/HIP conversion for finite values.
inline std::uint16_t f32_to_bf16(float x) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  const std::uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<std::uint16_t>(bits >> 16);
}

// Expand a bf16 bit pattern to fp32.
inline float bf16_to_f32(std::uint16_t b) {
  const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float x = 0.f;
  std::memcpy(&x, &bits, sizeof(x));
  return x;
}

// SplitMix64. The low bits select an expert; the same seed always yields the
// same expert.
inline std::uint64_t route_mix(std::uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}

// Scattered top-k destinations and the receive count of each rank.
inline CombineRoute build_route(int world, int tokens, int topk, int hidden,
                                int experts) {
  CombineRoute route;
  route.world = world;
  route.tokens = tokens;
  route.topk = topk;
  route.hidden = hidden;
  route.experts = experts;
  const std::size_t n =
      static_cast<std::size_t>(world) * tokens * topk;
  route.dest.assign(n, 0);
  route.slot.assign(n, 0);
  route.recv.assign(world, 0);
  const std::int64_t total_experts =
      static_cast<std::int64_t>(experts) * world;
  const int pool = static_cast<int>(total_experts);
  const int need = topk < pool ? topk : pool;
  std::vector<int> picked;
  picked.reserve(static_cast<std::size_t>(need));
  for (int src = 0; src < world; ++src) {
    for (int token = 0; token < tokens; ++token) {
      std::uint64_t state = route_mix(
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(src)) << 32) |
          static_cast<std::uint32_t>(token));
      picked.clear();
      while (static_cast<int>(picked.size()) < need) {
        state = route_mix(state + 0x9e3779b97f4a7c15ull);
        const int expert =
            static_cast<int>(state % static_cast<std::uint64_t>(pool));
        if (std::find(picked.begin(), picked.end(), expert) == picked.end())
          picked.push_back(expert);
      }
      for (int k = 0; k < topk; ++k) {
        const std::size_t id = route.index(src, token, k);
        // A top-k larger than the pool has no further expert to send.
        if (k >= static_cast<int>(picked.size())) {
          route.dest[id] = -1;
          route.slot[id] = -1;
          continue;
        }
        const int dest = picked[k] / experts;
        bool seen = false;
        for (int prev = 0; prev < k; ++prev)
          seen = seen || route.dest[route.index(src, token, prev)] == dest;
        // MORI dispatch keeps the earliest expert for a (token, rank) pair.
        if (seen) {
          route.dest[id] = -1;
          route.slot[id] = -1;
        } else {
          route.dest[id] = dest;
          route.slot[id] = route.recv[dest]++;
        }
      }
    }
  }
  return route;
}

// Slots this rank reads from other ranks.
inline std::size_t remote_slots(const CombineRoute &route, int rank) {
  std::size_t n = 0;
  for (int token = 0; token < route.tokens; ++token) {
    for (int k = 0; k < route.topk; ++k) {
      const int dest = route.dest[route.index(rank, token, k)];
      if (dest >= 0 && dest != rank)
        ++n;
    }
  }
  return n;
}

// Algo bytes for one rank: one hidden vector per kept token packed there.
inline double algo_bytes_of(const CombineRoute &route, int rank) {
  return static_cast<double>(route.recv[rank]) * route.hidden *
         sizeof(std::uint16_t);
}

// Fabric bytes for one rank: kept tokens that rank reads from other ranks.
inline double fabric_bytes_of(const CombineRoute &route, int rank) {
  return static_cast<double>(remote_slots(route, rank)) * route.hidden *
         sizeof(std::uint16_t);
}

// Rank whose time is reported. Byte counts reported with that time are this
// rank's. Equal times keep the rank with more fabric bytes, then more algo
// bytes, then the smaller index.
inline int slowest_rank(const double *ms, const double *fabric, const double *algo,
                        int world) {
  int slow = 0;
  for (int r = 1; r < world; ++r) {
    if (ms[r] > ms[slow])
      slow = r;
    else if (ms[r] == ms[slow] &&
             (fabric[r] > fabric[slow] ||
              (fabric[r] == fabric[slow] && algo[r] > algo[slow])))
      slow = r;
  }
  return slow;
}

// Largest absolute error over the first `checked` tokens and the last token.
inline double max_abs_error(const CombineRoute &route, int rank,
                            const std::uint16_t *got, int checked) {
  double error = 0.0;
  // Compare one token with the host reference. A non-finite difference is returned.
  auto consider = [&](int token) {
    for (int h = 0; h < route.hidden; ++h) {
      float acc = 0.f;
      for (int k = 0; k < route.topk; ++k) {
        const std::size_t id = route.index(rank, token, k);
        if (route.dest[id] < 0)
          continue;
        const float x = bf16_to_f32(f32_to_bf16(
            stage_unit(route.dest[id], route.slot[id], h)));
        acc += weight_of(token, k, route.topk) * x;
      }
      const float expect = bf16_to_f32(f32_to_bf16(acc));
      const float actual =
          bf16_to_f32(got[static_cast<std::size_t>(token) * route.hidden + h]);
      const double diff = std::fabs(static_cast<double>(actual) -
                                    static_cast<double>(expect));
      if (!std::isfinite(diff))
        return diff;
      error = std::max(error, diff);
    }
    return 0.0;
  };
  const int n = std::min(checked, route.tokens);
  for (int token = 0; token < n; ++token) {
    const double bad = consider(token);
    if (!std::isfinite(bad))
      return bad;
  }
  if (route.tokens > n) {
    const double bad = consider(route.tokens - 1);
    if (!std::isfinite(bad))
      return bad;
  }
  return error;
}
