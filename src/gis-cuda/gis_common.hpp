/*
 * Standalone benchmark derived from FBGEMM's sparse_group_index.cu
 * (group_index_select_or_add_2d_kernel; BSD license, Copyright Meta
 * Platforms, Inc. and affiliates).
 *
 * Forward (index select): for every table t in a group,
 *   output_t[r, :] = input_t[indices_t[r], :]
 * Backward (index add):
 *   input_grad_t[indices_t[r], :] += grad_output_t[r, :]   (atomic)
 *
 * Tables have different embedding dimensions (USE_VAR_COLS). Work is
 * distributed over logical warps of kWarpLanes lanes:
 *   - narrow tables (dim < kWarpLanes) pack kWarpLanes / dim rows into one
 *     warp so that lanes are not left idle (packed rows);
 *   - wide tables split a row into ceil(dim / kWarpLanes) column chunks.
 * Each logical warp ("worker") processes `steps` consecutive row groups of one
 * column chunk. In the backward pass every lane caches the partial sum for the
 * last destination row in a register and flushes it with one atomic add only
 * when the destination index changes, which coalesces consecutive updates to
 * the same row.
 *
 * Production FBGEMM does not have a MAX_RUN parameter. Its backward path can
 * sort indices (USE_SORTED_INDICES), after which a run length is the number of
 * lookups that selected that row. This benchmark instead generates synthetic
 * runs whose length is chosen uniformly from 1 through max_run. The default
 * max_run of 8 is not a measured production value.
 */
#pragma once

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GIS_HD __host__ __device__
#else
#define GIS_HD
#endif

#ifdef _OPENMP
#pragma omp declare target
#endif

constexpr int kWarpLanes = 32;      // lanes of a logical warp
constexpr int kThreadsPerBlock = 256;
constexpr int kForwardSteps = 1;    // one row group per worker, as in FBGEMM
constexpr int kBackwardSteps = 8;   // row groups per worker sharing a cache

struct GisLane {
  int table;
  int dim;
  int col;           // column handled by this lane
  int sub_row;       // row within a packed row group
  int rows_per_warp; // rows covered by one row group
  int first_group;   // first row group handled by the worker
  bool active;
};

// Rows covered by one logical warp. Narrow rows pack 32/dim rows into the warp;
// a row at least 32 columns wide occupies the whole warp, so this returns 1.
GIS_HD inline int gis_rows_per_warp(int dim) {
  return dim < kWarpLanes ? kWarpLanes / dim : 1;
}

// Number of 32-column tiles in one row. A narrow row uses one tile.
GIS_HD inline int gis_col_chunks(int dim) {
  return dim < kWarpLanes ? 1 : (dim + kWarpLanes - 1) / kWarpLanes;
}

// Map one lane of one logical warp to its table, column, and first row group.
// worker_offsets is the exclusive prefix sum of logical warps per table.
// steps is the number of consecutive row groups that warp will visit.
GIS_HD inline GisLane gis_locate(int64_t worker, int lane, int steps,
                                 const int64_t* worker_offsets,
                                 const int* dims, int tables) {
  // binary search: last table whose first worker is <= worker
  int lo = 0, hi = tables;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (worker_offsets[mid] <= worker) lo = mid; else hi = mid;
  }
  GisLane l;
  l.table = lo;
  l.dim = dims[lo];
  const int chunks = gis_col_chunks(l.dim);
  const int64_t member_worker = worker - worker_offsets[lo];
  const int chunk = member_worker % chunks;
  l.first_group = member_worker / chunks * steps;
  l.rows_per_warp = gis_rows_per_warp(l.dim);
  if (l.dim < kWarpLanes) {
    l.sub_row = lane / l.dim;
    l.col = lane - l.sub_row * l.dim;
    l.active = l.sub_row < l.rows_per_warp;
  } else {
    l.sub_row = 0;
    l.col = chunk * kWarpLanes + lane;
    l.active = l.col < l.dim;
  }
  return l;
}

#ifdef _OPENMP
#pragma omp end declare target
#endif

struct GisProblem {
  int tables;
  int rows;
  int work_rows;
  std::vector<int> dims;
  std::vector<int64_t> input_offsets;   // input / input_grad, rows x dim
  std::vector<int64_t> work_offsets;    // output / grad_output, work_rows x dim
  std::vector<int64_t> indices_offsets;
  std::vector<int64_t> forward_workers;
  std::vector<int64_t> backward_workers;
  std::vector<float> input;
  std::vector<float> grad_output;
  std::vector<int> indices;

  // Build tables whose widths cycle through dim*{1, 3/8, 1/8, 3/2, 1/4},
  // covering packed, single-warp, and multi-tile rows. max_run caps the
  // synthetic repeated-index runs; production run length is the per-row
  // lookup count after FBGEMM sorts the indices, and is not capped at 8.
  GisProblem(int table_count, int table_rows, int embedding_dim,
             int rows_to_select, int max_run)
      : tables(table_count), rows(table_rows), work_rows(rows_to_select),
        dims(table_count), input_offsets(table_count + 1, 0),
        work_offsets(table_count + 1, 0), indices_offsets(table_count + 1, 0),
        forward_workers(table_count + 1, 0),
        backward_workers(table_count + 1, 0) {
    // Variable embedding dimensions: narrow (packed), exactly one warp, and
    // wide (multi-chunk, including a partial chunk) tables.
    static const int scale[5] = {8, 3, 1, 12, 2};
    for (int t = 0; t < tables; ++t) {
      dims[t] = std::max(1, embedding_dim * scale[t % 5] / 8);
      const int64_t table_rows = rows;
      const int64_t selected_rows = work_rows;
      input_offsets[t + 1] = input_offsets[t] + table_rows * dims[t];
      work_offsets[t + 1] = work_offsets[t] + selected_rows * dims[t];
      indices_offsets[t + 1] = indices_offsets[t] + work_rows;
      const int64_t groups =
          (work_rows + gis_rows_per_warp(dims[t]) - 1) / gis_rows_per_warp(dims[t]);
      const int chunks = gis_col_chunks(dims[t]);
      forward_workers[t + 1] = forward_workers[t] +
          chunks * ((groups + kForwardSteps - 1) / kForwardSteps);
      backward_workers[t + 1] = backward_workers[t] +
          chunks * ((groups + kBackwardSteps - 1) / kBackwardSteps);
    }
    input.resize(input_offsets.back());
    grad_output.resize(work_offsets.back());
    indices.resize(indices_offsets.back());

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> values(-1.0f, 1.0f);
    std::uniform_int_distribution<int> row_dist(0, rows - 1);
    std::uniform_int_distribution<int> run_dist(1, max_run);
    for (float& v : input) v = values(rng);
    for (float& v : grad_output) v = values(rng);
    // Each run repeats one row index. Adjacent repeats are what the backward
    // register cache can coalesce. The length is uniform in [1, max_run].
    for (int t = 0; t < tables; ++t) {
      int r = 0;
      while (r < work_rows) {
        const int index = row_dist(rng);
        const int run = run_dist(rng);
        for (int k = 0; k < run && r < work_rows; ++k, ++r)
          indices[indices_offsets[t] + r] = index;
      }
    }
  }

  // Floats in the gathered output and in grad_output.
  int64_t work_elements() const { return work_offsets.back(); }
  // Floats in every embedding table and in their gradient buffers.
  int64_t input_elements() const { return input_offsets.back(); }
};

// Serial grouped index select: copy each selected embedding row to output.
inline void gis_forward_reference(const GisProblem& p, std::vector<float>& out) {
  out.assign(p.work_elements(), 0.0f);
  for (int t = 0; t < p.tables; ++t) {
    const int dim = p.dims[t];
    for (int64_t r = 0; r < p.work_rows; ++r) {
      const int64_t idx = p.indices[p.indices_offsets[t] + r];
      std::copy_n(p.input.data() + p.input_offsets[t] + idx * dim, dim,
                  out.data() + p.work_offsets[t] + r * dim);
    }
  }
}

// Serial grouped index add. Double precision makes the result independent of
// the order in which device atomics accumulate the same row.
inline void gis_backward_reference(const GisProblem& p, std::vector<float>& grad) {
  std::vector<double> acc(p.input_elements(), 0.0);
  for (int t = 0; t < p.tables; ++t) {
    const int dim = p.dims[t];
    for (int64_t r = 0; r < p.work_rows; ++r) {
      const int64_t idx = p.indices[p.indices_offsets[t] + r];
      const float* src = p.grad_output.data() + p.work_offsets[t] + r * dim;
      double* dst = acc.data() + p.input_offsets[t] + idx * dim;
      for (int c = 0; c < dim; ++c) dst[c] += src[c];
    }
  }
  grad.assign(acc.begin(), acc.end());
}

// Return true when every element is within atol + 1e-5*|expected|.
// Forward uses atol 0; backward uses 1e-3 to allow atomic rounding.
inline bool gis_check(const std::vector<float>& actual,
                      const std::vector<float>& expected, float atol,
                      const char* name) {
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!(std::fabs(actual[i] - expected[i]) <=
          atol + 1e-5f * std::fabs(expected[i]))) {
      std::printf("%s mismatch at %zu: %.8f != %.8f\n", name, i, actual[i],
                  expected[i]);
      return false;
    }
  }
  return true;
}

// Parse the CLI. The optional sixth value is the synthetic maximum index-run
// length. It defaults to 8 and is not a value reported for production.
inline bool gis_parse_args(int argc, char** argv, int& tables, int& rows,
                           int& dim, int& work_rows, int& repeat,
                           int& max_run) {
  if (argc < 6 || argc > 7) {
    std::printf("Usage: %s <tables> <rows/table> <embedding-dim> "
                "<selected-rows/table> <repeat> [max-index-run-length]\n",
                argv[0]);
    return false;
  }
  tables = std::atoi(argv[1]);
  rows = std::atoi(argv[2]);
  dim = std::atoi(argv[3]);
  work_rows = std::atoi(argv[4]);
  repeat = std::atoi(argv[5]);
  max_run = argc == 7 ? std::atoi(argv[6]) : 8;
  if (tables <= 0 || rows <= 0 || dim <= 0 || work_rows <= 0 || repeat <= 0 ||
      max_run <= 0) {
    std::printf("All arguments must be positive.\n");
    return false;
  }
  return true;
}

// Print the generated per-table widths and logical-warp counts.
inline void gis_print_problem(const GisProblem& p) {
  std::printf("tables=%d rows/table=%d selected rows/table=%d dims=", p.tables,
              p.rows, p.work_rows);
  for (int t = 0; t < p.tables; ++t)
    std::printf("%d%s", p.dims[t], t + 1 < p.tables ? "," : "\n");
  std::printf("forward workers=%" PRId64 " backward workers=%" PRId64 "\n",
              p.forward_workers.back(), p.backward_workers.back());
}
