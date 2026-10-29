// Standalone table-batched embedding benchmark support.
//
// The workload and GPU mapping are derived from the public FBGEMM
// SplitTableBatchedEmbeddingBags implementation:
// https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu
// FBGEMM is distributed under the BSD 3-Clause license:
// https://github.com/pytorch/FBGEMM/blob/main/LICENSE
// This is an independent, dependency-free implementation; no FBGEMM or
// PyTorch source is copied here.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// In-source defaults are a small smoke shape. Production sizes live in the
// Makefiles: 5 tables, batch 65536, 1000000 rows, D=192, bag size 55.
// Bag size is how many rows are summed into one output. A production table
// has no separate cap on how many lookups select the same row.
struct TBEProblem {
  int tables = 8;
  int batch = 128;
  int rows = 4096;
  int dim = 128;
  int bag = 20;
  int repeat = 20;

  // Embedding weights: tables * rows_per_table * embedding_dim.
  size_t weight_count() const {
    const size_t ntables = tables;
    const size_t nrows = rows;
    return ntables * nrows * dim;
  }
  // Lookup indices: one bag of `bag` rows for every table and batch row.
  size_t index_count() const {
    const size_t ntables = tables;
    const size_t nbatch = batch;
    return ntables * nbatch * bag;
  }
  // Pooled outputs: one embedding per table and batch row.
  size_t output_count() const {
    const size_t ntables = tables;
    const size_t nbatch = batch;
    return ntables * nbatch * dim;
  }
  // Pooled rows assigned to warps, subgroups, or OpenMP teams.
  int bags() const { return tables * batch; }
};

// 256-thread blocks. One warp, wavefront, or subgroup owns one pooled row.
constexpr int tbe_block_threads = 256;

// `lanes` is the warp, wave, or subgroup size queried on the host.
inline int tbe_grid_blocks(int bags, int lanes) {
  const int waves = tbe_block_threads / lanes;
  return (bags + waves - 1) / waves;
}

// Print the six positional arguments. Omitting them keeps the defaults above.
inline void tbe_usage(const char* name) {
  std::printf(
      "Usage: %s [tables batch rows_per_table embedding_dim bag_size repeat]\n",
      name);
}

// Read tables, batch, rows per table, embedding dimension, bag size, and repeat.
inline bool tbe_parse(int argc, char** argv, TBEProblem& p) {
  if (argc != 1 && argc != 7) {
    tbe_usage(argv[0]);
    return false;
  }
  if (argc == 7) {
    p.tables = std::atoi(argv[1]);
    p.batch = std::atoi(argv[2]);
    p.rows = std::atoi(argv[3]);
    p.dim = std::atoi(argv[4]);
    p.bag = std::atoi(argv[5]);
    p.repeat = std::atoi(argv[6]);
  }
  if (p.tables <= 0 || p.batch <= 0 || p.rows <= 0 || p.dim <= 0 ||
      p.bag <= 0 || p.repeat <= 0) {
    std::fprintf(stderr, "All arguments must be positive.\n");
    return false;
  }
  return true;
}

// FP16 storage for the Table II production precision (arXiv:2610.22230).
// Bag sums accumulate in FP32 and round back to FP16.
struct TbeHalf {
  uint16_t bits;
};

inline uint32_t tbe_float_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float tbe_bits_to_float(uint32_t bits) {
  float value = 0.f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Round-to-nearest-even binary16, including subnormals.
inline TbeHalf tbe_float_to_half(float value) {
  const uint32_t bits = tbe_float_bits(value);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t mag = bits & 0x7fffffffu;
  uint16_t out = 0;
  if (mag > 0x7f800000u) {
    const uint32_t nan = mag & 0x7fffffu;
    out = uint16_t(0x7e00u | (nan >> 13));
    if ((out & 0x03ffu) == 0) out |= 1u;
  } else if (mag >= 0x47800000u) {
    out = 0x7c00u;
  } else if (mag < 0x38800000u) {
    const float scaled = tbe_bits_to_float(mag) + tbe_bits_to_float(0x38800000u);
    out = uint16_t(tbe_float_bits(scaled) - 0x38800000u);
  } else {
    const uint32_t mant_odd = (mag >> 13) & 1u;
    const uint32_t rounded = mag - ((127u - 15u) << 23) + 0xfffu + mant_odd;
    out = uint16_t(rounded >> 13);
  }
  return TbeHalf{uint16_t(sign | out)};
}

inline float tbe_half_to_float(TbeHalf value) {
  const uint32_t bits = value.bits;
  const uint32_t sign = (bits & 0x8000u) << 16;
  const uint32_t expo = (bits >> 10) & 0x1fu;
  const uint32_t mant = bits & 0x03ffu;
  uint32_t out = 0;
  if (expo == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      uint32_t m = mant;
      int32_t e = -1;
      do {
        e++;
        m <<= 1;
      } while ((m & 0x0400u) == 0);
      m &= 0x03ffu;
      out = sign | (uint32_t(127 - 15 + 1 - e) << 23) | (m << 13);
    }
  } else if (expo == 31) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    out = sign | ((expo + (127 - 15)) << 23) | (mant << 13);
  }
  return tbe_bits_to_float(out);
}

// Map a counter to a deterministic value in [-1, 1). The hash stays unsigned;
// its low 11 bits become a signed offset before scaling so the range is centered.
inline float tbe_value(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  const int bin = int(x & 2047u) - 1024;
  return float(bin) / 1024.0f;
}

// Fill weights, lookup indices, and output gradients.
// Indices are a fixed mix modulo the row count, so collisions are expected
// and the backward kernel must accumulate them.
inline void tbe_initialize(
    const TBEProblem& p,
    std::vector<TbeHalf>& weights,
    std::vector<int>& indices,
    std::vector<TbeHalf>& grad_output) {
  weights.resize(p.weight_count());
  indices.resize(p.index_count());
  grad_output.resize(p.output_count());
  for (size_t i = 0; i < weights.size(); ++i)
    weights[i] = tbe_float_to_half(tbe_value(uint32_t(i + 17)));
  for (int t = 0; t < p.tables; ++t) {
    const size_t table = t;
    for (int b = 0; b < p.batch; ++b) {
      const size_t batch_row = b;
      for (int l = 0; l < p.bag; ++l) {
        const size_t i = (table * p.batch + batch_row) * p.bag + l;
        const uint64_t mixed = uint64_t(t + 1) * 2654435761ull +
                               uint64_t(b) * 97ull + uint64_t(l) * 13ull;
        indices[i] = mixed % uint64_t(p.rows);
      }
    }
  }
  for (size_t i = 0; i < grad_output.size(); ++i)
    grad_output[i] = tbe_float_to_half(0.25f * tbe_value(uint32_t(i + 991)));
}

// output[t, b, :] = sum of weights[t, index, :] over the bag.
inline void tbe_forward_reference(
    const TBEProblem& p,
    const TbeHalf* weights,
    const int* indices,
    TbeHalf* output) {
  std::fill(output, output + p.output_count(), TbeHalf{0});
  for (int t = 0; t < p.tables; ++t) {
    const size_t table = t;
    for (int b = 0; b < p.batch; ++b) {
      const size_t batch_row = b;
      const size_t o = (table * p.batch + batch_row) * p.dim;
      for (int d = 0; d < p.dim; ++d) {
        float sum = 0.f;
        for (int l = 0; l < p.bag; ++l) {
          const int r = indices[(table * p.batch + batch_row) * p.bag + l];
          const size_t row = r;
          const size_t w = (table * p.rows + row) * p.dim;
          sum += tbe_half_to_float(weights[w + d]);
        }
        output[o + d] = tbe_float_to_half(sum);
      }
    }
  }
}

// grad_weight[t, index, :] += grad_output[t, b, :] for every bag entry.
inline void tbe_backward_reference(
    const TBEProblem& p,
    const int* indices,
    const TbeHalf* grad_output,
    TbeHalf* grad_weight) {
  std::fill(grad_weight, grad_weight + p.weight_count(), TbeHalf{0});
  for (int t = 0; t < p.tables; ++t) {
    const size_t table = t;
    for (int b = 0; b < p.batch; ++b) {
      const size_t batch_row = b;
      const size_t g = (table * p.batch + batch_row) * p.dim;
      for (int l = 0; l < p.bag; ++l) {
        const int r = indices[(table * p.batch + batch_row) * p.bag + l];
        const size_t row = r;
        const size_t w = (table * p.rows + row) * p.dim;
        for (int d = 0; d < p.dim; ++d) {
          const float sum = tbe_half_to_float(grad_weight[w + d]) +
                            tbe_half_to_float(grad_output[g + d]);
          grad_weight[w + d] = tbe_float_to_half(sum);
        }
      }
    }
  }
}

// Compare one kernel result with the reference. The printed PASS is this call.
inline bool tbe_check(
    const char* label,
    const std::vector<TbeHalf>& got,
    const std::vector<TbeHalf>& ref,
    float atol = 5.0e-2f) {
  float max_error = 0.0f;
  size_t worst = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float e = std::fabs(tbe_half_to_float(got[i]) - tbe_half_to_float(ref[i]));
    if (e > max_error) {
      max_error = e;
      worst = i;
    }
  }
  const bool ok = max_error <= atol;
  std::printf(
      "%s validation: %s (max abs error %.6g at %zu)\n",
      label,
      ok ? "PASS" : "FAIL",
      max_error,
      worst);
  return ok;
}

// Print the shape that will be timed. Makefile `run` overrides the defaults.
inline void tbe_print_problem(const TBEProblem& p, const char* backend) {
  std::printf(
      "TBE %s: tables=%d batch=%d rows=%d D=%d bag=%d repeat=%d prec=fp16\n",
      backend,
      p.tables,
      p.batch,
      p.rows,
      p.dim,
      p.bag,
      p.repeat);
}
