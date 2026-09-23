#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <tuple>
#include <unordered_map>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "mace_nlist.hpp"
#include "mace_pipeline.hpp"

using namespace mace_pipeline;

#define HIP_CHECK(call)                                                         \
  do {                                                                          \
    hipError_t status_ = (call);                                                 \
    if (status_ != hipSuccess)                                                   \
      throw std::runtime_error(std::string(#call) + ": " +                       \
                               hipGetErrorString(status_));                      \
  } while (0)

#define ROCBLAS_CHECK(call)                                                     \
  do {                                                                          \
    rocblas_status status_ = (call);                                             \
    if (status_ != rocblas_status_success)                                       \
      throw std::runtime_error(std::string(#call) +                              \
                               ": rocBLAS status " +                             \
                               std::to_string(static_cast<int>(status_)));       \
  } while (0)

constexpr int kBlock = 256;

struct BlasContext {
  rocblas_handle handle = nullptr;
  BlasContext() {
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
  }
  ~BlasContext() { if (handle) (void)rocblas_destroy_handle(handle); }
  BlasContext(const BlasContext &) = delete;
  BlasContext &operator=(const BlasContext &) = delete;
};

struct Buffer {
  float *data = nullptr;
  std::size_t capacity = 0;
  ~Buffer() { if (data) (void)hipFree(data); }
  void ensure(std::size_t count) {
    if (count <= capacity) return;
    if (data) HIP_CHECK(hipFree(data));
    HIP_CHECK(hipMalloc(&data, count * sizeof(float)));
    capacity = count;
  }
};

template <class T>
struct TypedBuffer {
  T *data = nullptr;
  std::size_t capacity = 0;
  ~TypedBuffer() { if (data) (void)hipFree(data); }
  void ensure(std::size_t count) {
    if (count <= capacity) return;
    if (data) HIP_CHECK(hipFree(data));
    HIP_CHECK(hipMalloc(&data, count * sizeof(T)));
    capacity = count;
  }
};

constexpr int kMaxNeighbors = 20;
// One wavefront on CUDA's sm_52 measured best at 64 lanes; the wider LDS and
// 64-wide wavefront on gfx908 move the optimum to 128. Only the constant
// differs from the CUDA reference.
constexpr int kNlistBlock = 64;
constexpr double kCutoff = 5.0;

struct Candidate {
  double r2;
  int src, sx, sy, sz;
};

__device__ bool better(const Candidate &a, const Candidate &b) {
  if (a.r2 != b.r2) return a.r2 < b.r2;
  if (a.src != b.src) return a.src < b.src;
  if (a.sx != b.sx) return a.sx < b.sx;
  if (a.sy != b.sy) return a.sy < b.sy;
  return a.sz < b.sz;
}

// Staging capacity for candidates that passed the cutoff but are not yet
// merged into the running top-20. One image round appends at most one
// candidate per lane, so draining once the pool is within a round of the
// limit keeps a round from ever overrunning it.
constexpr int kNlistPool = 512;
constexpr int kNlistPoolLimit = kNlistPool - kMaxNeighbors - kNlistBlock;
static_assert(kNlistPoolLimit > 0, "pool must hold one round plus the top-20");

// Collapses the staged pool and the running top-20 into a fresh top-20.
// `better` is a strict total order on the distinct (r2, src, sx, sy, sz)
// tuples, so ranking every pool element against every other reproduces the
// serial insertion result exactly, independent of the order in which lanes
// appended their candidates. Every thread of the block must take part.
__device__ void merge_candidates(Candidate *pool, int *pool_count,
                                 Candidate *best, int *best_count,
                                 Candidate *ranked) {
  __syncthreads();
  const int staged = *pool_count;
  const int kept = *best_count;
  for (int index = threadIdx.x; index < kept; index += kNlistBlock)
    pool[staged + index] = best[index];
  __syncthreads();
  const int total = staged + kept;
  for (int index = threadIdx.x; index < total; index += kNlistBlock) {
    const Candidate candidate = pool[index];
    int rank = 0;
    for (int other = 0; other < total; ++other)
      rank += better(pool[other], candidate) ? 1 : 0;
    if (rank < kMaxNeighbors) ranked[rank] = candidate;
  }
  __syncthreads();
  const int keep = total < kMaxNeighbors ? total : kMaxNeighbors;
  for (int index = threadIdx.x; index < keep; index += kNlistBlock)
    best[index] = ranked[index];
  if (threadIdx.x == 0) {
    *best_count = keep;
    *pool_count = 0;
  }
  __syncthreads();
}

__device__ double directed_nextafter(double value, bool positive) {
  if (value == 0.0)
    return __longlong_as_double(
        positive ? 1LL : static_cast<long long>(0x8000000000000001ULL));
  long long bits = __double_as_longlong(value);
  bits += ((value > 0.0) == positive) ? 1 : -1;
  return __longlong_as_double(bits);
}

__global__ void nlist_kernel(
    std::size_t nodes, const std::uint64_t *graph_ptr,
    const std::uint64_t *node_batch, const double *positions,
    const double *cells, const double *inverses, const std::uint64_t *pbc,
    Candidate *slots, int *counts) {
  const int dst = blockIdx.x;
  if (static_cast<std::size_t>(dst) >= nodes) return;
  const int lane = threadIdx.x;
  const int graph = static_cast<int>(node_batch[dst]);
  const double *h = cells + 9 * graph;
  const double *iv = inverses + 9 * graph;
  const double bx = kCutoff * sqrt(iv[0] * iv[0] + iv[3] * iv[3] + iv[6] * iv[6]);
  const double by = kCutoff * sqrt(iv[1] * iv[1] + iv[4] * iv[4] + iv[7] * iv[7]);
  const double bz = kCutoff * sqrt(iv[2] * iv[2] + iv[5] * iv[5] + iv[8] * iv[8]);
  __shared__ Candidate pool[kNlistPool];
  __shared__ Candidate best[kMaxNeighbors];
  __shared__ Candidate ranked[kMaxNeighbors];
  __shared__ int max_images;
  __shared__ int pool_count;
  __shared__ int count;
  if (lane == 0) {
    count = 0;
    pool_count = 0;
  }
  __syncthreads();

  const int first_src = static_cast<int>(graph_ptr[graph]);
  const int last_src = static_cast<int>(graph_ptr[graph + 1]);
  for (int base = first_src; base < last_src; base += kNlistBlock) {
    const int src = base + lane;
    double x = 0.0, y = 0.0, z = 0.0;
    int lx = 0, ux = -1, ly = 0, uy = -1, lz = 0, uz = -1;
    int image_count = 0;
    if (src < last_src) {
      x = positions[3 * dst] - positions[3 * src];
      y = positions[3 * dst + 1] - positions[3 * src + 1];
      z = positions[3 * dst + 2] - positions[3 * src + 2];
      const double fx = x * iv[0] + y * iv[3] + z * iv[6];
      const double fy = x * iv[1] + y * iv[4] + z * iv[7];
      const double fz = x * iv[2] + y * iv[5] + z * iv[8];
      lx = pbc[3 * graph] ? static_cast<int>(ceil(directed_nextafter(-fx - bx, false))) : 0;
      ux = pbc[3 * graph] ? static_cast<int>(floor(directed_nextafter(-fx + bx, true))) : 0;
      ly = pbc[3 * graph + 1] ? static_cast<int>(ceil(directed_nextafter(-fy - by, false))) : 0;
      uy = pbc[3 * graph + 1] ? static_cast<int>(floor(directed_nextafter(-fy + by, true))) : 0;
      lz = pbc[3 * graph + 2] ? static_cast<int>(ceil(directed_nextafter(-fz - bz, false))) : 0;
      uz = pbc[3 * graph + 2] ? static_cast<int>(floor(directed_nextafter(-fz + bz, true))) : 0;
      if (lx <= ux && ly <= uy && lz <= uz)
        image_count = (ux - lx + 1) * (uy - ly + 1) * (uz - lz + 1);
    }
    if (lane == 0) max_images = 0;
    __syncthreads();
    atomicMax(&max_images, image_count);
    __syncthreads();

    const int nz = uz - lz + 1;
    const int ny = uy - ly + 1;
    for (int image = 0; image < max_images; ++image) {
      if (image < image_count) {
        const int sx = lx + image / (ny * nz);
        const int remainder = image % (ny * nz);
        const int sy = ly + remainder / nz;
        const int sz = lz + remainder % nz;
        const double dx = x + sx * h[0] + sy * h[3] + sz * h[6];
        const double dy = y + sx * h[1] + sy * h[4] + sz * h[7];
        const double dz = z + sx * h[2] + sy * h[5] + sz * h[8];
        Candidate c;
        c.r2 = dx * dx + dy * dy + dz * dz;
        c.src = src; c.sx = sx; c.sy = sy; c.sz = sz;
        if (c.r2 <= kCutoff * kCutoff &&
            !(src == dst && sx == 0 && sy == 0 && sz == 0))
          pool[atomicAdd(&pool_count, 1)] = c;
      }
      __syncthreads();
      if (pool_count > kNlistPoolLimit)
        merge_candidates(pool, &pool_count, best, &count, ranked);
    }
  }
  merge_candidates(pool, &pool_count, best, &count, ranked);
  if (lane == 0) counts[dst] = count;
  for (int index = lane; index < count; index += kNlistBlock)
    slots[dst * kMaxNeighbors + index] = best[index];
}

// Hierarchical exclusive scan over an arbitrary element count: every level
// scans within a block and hands its block totals to the next level, so the
// number of levels grows with log(nodes) instead of imposing a ceiling. All
// arithmetic is exact 64-bit integer addition, so the result is deterministic.
constexpr int kScanBlock = 256;

__device__ void block_exclusive_scan(std::uint64_t *shared,
                                     std::uint64_t value,
                                     std::uint64_t &exclusive,
                                     std::uint64_t &total) {
  const int lane = threadIdx.x;
  shared[lane] = value;
  __syncthreads();
  for (int offset = 1; offset < kScanBlock; offset <<= 1) {
    const std::uint64_t addend = lane >= offset ? shared[lane - offset] : 0;
    __syncthreads();
    shared[lane] += addend;
    __syncthreads();
  }
  exclusive = shared[lane] - value;
  total = shared[kScanBlock - 1];
  __syncthreads();
}

__global__ void scan_counts_kernel(const int *counts, std::size_t nodes,
                                   std::uint64_t *offsets,
                                   std::uint64_t *block_totals,
                                   std::size_t length) {
  __shared__ std::uint64_t shared[kScanBlock];
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * kScanBlock + threadIdx.x;
  const std::uint64_t value =
      i < nodes ? static_cast<std::uint64_t>(counts[i]) : 0;
  std::uint64_t exclusive = 0, total = 0;
  block_exclusive_scan(shared, value, exclusive, total);
  if (i < length) offsets[i] = exclusive;
  if (threadIdx.x == 0) block_totals[blockIdx.x] = total;
}

__global__ void scan_values_kernel(std::uint64_t *values,
                                   std::uint64_t *block_totals,
                                   std::size_t length) {
  __shared__ std::uint64_t shared[kScanBlock];
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * kScanBlock + threadIdx.x;
  const std::uint64_t value = i < length ? values[i] : 0;
  std::uint64_t exclusive = 0, total = 0;
  block_exclusive_scan(shared, value, exclusive, total);
  if (i < length) values[i] = exclusive;
  if (threadIdx.x == 0) block_totals[blockIdx.x] = total;
}

__global__ void scan_offset_kernel(std::uint64_t *values,
                                   const std::uint64_t *block_offsets,
                                   std::size_t length) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * kScanBlock + threadIdx.x;
  if (i < length) values[i] += block_offsets[blockIdx.x];
}

__global__ void flatten_kernel(
    std::size_t nodes, const std::uint64_t *node_batch, const double *cells,
    const int *counts, const std::uint64_t *offsets, const Candidate *slots,
    std::uint64_t *edges, double *shifts) {
  const std::size_t dst = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst >= nodes) return;
  const double *h = cells + 9 * node_batch[dst];
  for (int j = 0; j < counts[dst]; ++j) {
    const Candidate &c = slots[dst * kMaxNeighbors + j];
    const std::size_t e = offsets[dst] + j;
    edges[2 * e] = c.src;
    edges[2 * e + 1] = dst;
    shifts[3 * e] = c.sx * h[0] + c.sy * h[3] + c.sz * h[6];
    shifts[3 * e + 1] = c.sx * h[1] + c.sy * h[4] + c.sz * h[7];
    shifts[3 * e + 2] = c.sx * h[2] + c.sy * h[5] + c.sz * h[8];
  }
}

__global__ void convert_shifts_kernel(const double *input, float *output,
                                      std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = static_cast<float>(input[i]);
}

struct DeviceTensor {
  float *f32 = nullptr;
  double *f64 = nullptr;
  std::uint64_t *u64 = nullptr;
  ~DeviceTensor() {
    if (f32) (void)hipFree(f32);
    if (f64) (void)hipFree(f64);
    if (u64) (void)hipFree(u64);
  }
};

struct DeviceFixture {
  std::unordered_map<std::string, std::unique_ptr<DeviceTensor>> tensors;
  explicit DeviceFixture(const Fixture &fixture) {
    for (const auto &entry : fixture.tensors) {
      if (entry.first.rfind("expected_", 0) == 0 &&
          entry.first != "expected_node_attributes")
        continue;
      if (entry.first == "edge_index" || entry.first == "edge_shifts" ||
          entry.first == "edge_attr_raw")
        continue;  // Captured topology is validation-only.
      auto tensor = std::make_unique<DeviceTensor>();
      if (!entry.second.f64.empty()) {
        std::vector<float> values(entry.second.f64.begin(),
                                  entry.second.f64.end());
        HIP_CHECK(hipMalloc(&tensor->f32, values.size() * sizeof(float)));
        HIP_CHECK(hipMemcpy(tensor->f32, values.data(),
                            values.size() * sizeof(float),
                            hipMemcpyHostToDevice));
        if (entry.first == "positions" || entry.first == "cells") {
          HIP_CHECK(hipMalloc(&tensor->f64,
                              entry.second.f64.size() * sizeof(double)));
          HIP_CHECK(hipMemcpy(tensor->f64, entry.second.f64.data(),
                              entry.second.f64.size() * sizeof(double),
                              hipMemcpyHostToDevice));
        }
      } else {
        HIP_CHECK(hipMalloc(&tensor->u64,
                            entry.second.u64.size() * sizeof(std::uint64_t)));
        HIP_CHECK(hipMemcpy(tensor->u64, entry.second.u64.data(),
                            entry.second.u64.size() * sizeof(std::uint64_t),
                            hipMemcpyHostToDevice));
      }
      tensors.emplace(entry.first, std::move(tensor));
    }
  }
  const DeviceTensor &at(const std::string &name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("missing device tensor: " + name);
    return *it->second;
  }
};

__global__ void one_hot_kernel(const float *raw, float *output,
                               std::size_t nodes) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nodes * kSpecies) return;
  const std::size_t n = i / kSpecies;
  output[i] = i % kSpecies == static_cast<std::size_t>(llrint(raw[n]) - 1)
                  ? 1.0f : 0.0f;
}

__global__ void embedding_gather_kernel(const std::uint8_t *species,
                                        const float *matrix, float *output,
                                        std::size_t nodes) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nodes * kChannels) return;
  const std::size_t n = i / kChannels, c = i % kChannels;
  output[i] = matrix[static_cast<std::size_t>(species[n]) * kChannels + c];
}

__global__ void edge_kernel(const float *positions, const std::uint64_t *edges,
                            const float *shifts, const float *bessel,
                            const float *constants,
                            float *attributes, float *features,
                            std::size_t edge_count) {
  const std::size_t e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= edge_count) return;
  const std::size_t sender = edges[2 * e];
  const std::size_t receiver = edges[2 * e + 1];
  float vector[3], r2 = 0.0f;
  for (int m = 0; m < 3; ++m) {
    vector[m] = positions[3 * receiver + m] - positions[3 * sender + m] +
                shifts[3 * e + m];
    r2 += vector[m] * vector[m];
  }
  const float r = sqrtf(r2);
  attributes[5 * e] = r;
  attributes[5 * e + 1] = 1.0f;
  for (int m = 0; m < 3; ++m)
    attributes[5 * e + 2 + m] = sqrtf(3.0f) * vector[m] / r;
  const float p = constants[2], q = r / constants[1];
  const float qp = powf(q, p);
  const float qp1 = qp * q;
  const float qp2 = qp1 * q;
  const float cutoff =
      r < constants[1]
          ? 1.0f - ((p + 1.0f) * (p + 2.0f) / 2.0f) * qp +
                p * (p + 2.0f) * qp1 -
                (p * (p + 1.0f) / 2.0f) * qp2
          : 0.0f;
  for (int k = 0; k < 6; ++k)
    features[6 * e + k] = constants[0] * sinf(bessel[k] * r) / r * cutoff;
}

__global__ void condition_kernel(const float *scalar, const std::uint64_t *batch,
                                 const float *graph, const float *weight,
                                 const float *bias, float *result,
                                 std::size_t nodes) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nodes * kChannels) return;
  const std::size_t n = i / kChannels, out = i % kChannels;
  float sum = bias[out];
  for (std::size_t in = 0; in < kChannels; ++in)
    sum += scalar[n * kChannels + in] * weight[in * kChannels + out];
  result[i] = sum + graph[batch[n]] * weight[kChannels * kChannels + out];
}

__global__ void combine_kernel(const float *scalar, const float *equivariant,
                               float *input, std::size_t nodes, bool vectors) {
  const std::size_t width = vectors ? 512 : 128;
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nodes * width) return;
  const std::size_t n = i / width, c = i % width;
  input[i] = c < 128 ? scalar[n * 128 + c]
                     : equivariant[n * 384 + c - 128];
}

__global__ void fc_input_kernel(const float *edge_features, const float *down,
                                const std::uint64_t *edges, float *current,
                                std::size_t edge_count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= edge_count * 262) return;
  const std::size_t e = i / 262, c = i % 262;
  if (c < 6) current[i] = edge_features[6 * e + c];
  else if (c < 134) current[i] = down[edges[2 * e] * 128 + c - 6];
  else current[i] = down[edges[2 * e + 1] * 128 + c - 134];
}

__global__ void silu_kernel(float *values, std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const float x = values[i];
    values[i] = static_cast<float>(kSiluScale) * x / (1.0f + expf(-x));
  }
}

__device__ float tensor_product_value(
    int layer, std::size_t e, std::size_t o, const std::uint64_t *edges,
    const float *up, const float *attrs, const float *weights) {
  const std::size_t sender = edges[2 * e];
  const float *x = up + sender * (layer == 0 ? 128 : 512);
  const float *a = attrs + 5 * e;
  const float *w = weights + e * (layer == 0 ? 384 : 768);
  float value = 0.0f;
  if (o < 128) {
    value = x[o] * (w[2 * o] * a[0] + w[2 * o + 1] * a[1]) / sqrtf(2.0f);
  } else if (layer && o < 256) {
    const std::size_t c = o - 128;
    float dot = 0.0f;
    for (int m = 0; m < 3; ++m) dot += x[128 + 3 * c + m] * a[2 + m];
    value = dot * w[256 + c] / sqrtf(3.0f);
  } else {
    const std::size_t base = layer == 0 ? 128 : 256;
    if (o >= base && o < base + 384) {
      const std::size_t c = (o - base) / 3, m = (o - base) % 3;
      value = x[c] * a[2 + m] * w[(layer == 0 ? 256 : 384) + c];
    } else if (layer && o >= 640) {
      const std::size_t c = (o - 640) / 3, m = (o - 640) % 3;
      value = x[128 + 3 * c + m] *
              (w[512 + 2 * c] * a[0] + w[513 + 2 * c] * a[1]) / sqrtf(2.0f);
    }
  }
  return value;
}

__global__ void tensor_product_kernel(
    int layer, const std::uint64_t *edges, const float *up, const float *attrs,
    const float *weights, float *messages, std::size_t edge_count) {
  const std::size_t width = layer == 0 ? 512 : 1024;
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= edge_count * width) return;
  messages[i] = tensor_product_value(
      layer, i / width, i % width, edges, up, attrs, weights);
}

__global__ void aggregate_kernel(const std::uint64_t *neighbor_offsets,
                                 const float *messages, float *output,
                                 std::size_t nodes, std::size_t width) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nodes * width) return;
  const std::size_t n = i / width, c = i % width;
  float sum = 0.0f;
  for (std::size_t e = neighbor_offsets[n]; e < neighbor_offsets[n + 1]; ++e)
    sum += messages[e * width + c];
  output[i] = sum;
}

__global__ void tensor_product_aggregate_kernel(
    int layer, const std::uint64_t *edges,
    const std::uint64_t *neighbor_offsets, const float *up,
    const float *attrs, const float *weights, float *output,
    std::size_t nodes) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nodes * kChannels) return;
  const std::size_t n = i / kChannels, c = i % kChannels;
  const std::size_t width = layer == 0 ? 512 : 1024;
  const std::size_t input_width = layer == 0 ? 128 : 512;
  float scalar = 0.0f, contracted = 0.0f;
  float vectors[3] = {0.0f, 0.0f, 0.0f};
  float carried[3] = {0.0f, 0.0f, 0.0f};
  constexpr float inv_root2 = 0.70710678118654752440f;
  constexpr float inv_root3 = 0.57735026918962576451f;
  for (std::size_t e = neighbor_offsets[n]; e < neighbor_offsets[n + 1]; ++e) {
    const float *x = up + edges[2 * e] * input_width;
    const float *a = attrs + 5 * e;
    const float *w = weights + e * (layer == 0 ? 384 : 768);
    scalar += x[c] * (w[2 * c] * a[0] + w[2 * c + 1] * a[1]) * inv_root2;
    const float vector_scale = x[c] * w[(layer == 0 ? 256 : 384) + c];
    for (int m = 0; m < 3; ++m) vectors[m] += vector_scale * a[2 + m];
    if (layer) {
      float dot = 0.0f;
      for (int m = 0; m < 3; ++m)
        dot += x[128 + 3 * c + m] * a[2 + m];
      contracted += dot * w[256 + c] * inv_root3;
      const float carry_scale =
          (w[512 + 2 * c] * a[0] + w[513 + 2 * c] * a[1]) * inv_root2;
      for (int m = 0; m < 3; ++m)
        carried[m] += x[128 + 3 * c + m] * carry_scale;
    }
  }
  output[n * width + c] = scalar;
  const std::size_t vector_out = layer == 0 ? 128 + 3 * c : 256 + 3 * c;
  for (int m = 0; m < 3; ++m)
    output[n * width + vector_out + m] = vectors[m];
  if (layer) {
    output[n * width + 128 + c] = contracted;
    for (int m = 0; m < 3; ++m)
      output[n * width + 640 + 3 * c + m] = carried[m];
  }
}

__global__ void scale_kernel(float *values, std::size_t count,
                             const float *constants) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) values[i] /= constants[3];
}

__global__ void reshape_kernel(const float *post, float *output,
                               std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::size_t n = i / 512, c4 = i % 512;
  const std::size_t c = c4 / 4, m = c4 % 4;
  output[i] = m == 0 ? post[n * 512 + c]
                     : post[n * 512 + 128 + 3 * c + m - 1];
}

__global__ void scale_reshape_kernel(const float *post, float *output,
                                     std::size_t count, float divisor) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::size_t n = i / 512, c4 = i % 512;
  const std::size_t c = c4 / 4, m = c4 % 4;
  const std::size_t source =
      m == 0 ? n * 512 + c : n * 512 + 128 + 3 * c + m - 1;
  output[i] = post[source] / divisor;
}

__global__ void sym_kernel(
    int layer, const float *input, const std::uint8_t *node_species, const float *sw1,
    const float *sw2, const float *su1, const float *su2, const float *vw1,
    const float *vw2, const float *vu1, const float *vu2, float *output,
    std::size_t nodes) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nodes * kChannels) return;
  const bool vectors = layer < 3;
  const std::size_t width = vectors ? 512 : 128;
  const std::size_t n = i / kChannels, c = i % kChannels;
  const std::size_t species = node_species[n];
  const float *x = input + (n * kChannels + c) * 4;
  const float scalar_w1 = sw1[species * 128 + c];
  const float scalar_w20 = sw2[(species * 2) * 128 + c];
  const float scalar_w21 = sw2[(species * 2 + 1) * 128 + c];
  float products[16];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b) products[4 * a + b] = x[a] * x[b];
  float scalar = 0.0f;
  for (int a = 0; a < 4; ++a)
    scalar += su1[a] * scalar_w1 * x[a];
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 4; ++b) {
      const float product = products[4 * a + b];
      scalar += su2[(a * 4 + b) * 2] * scalar_w20 * product;
      scalar += su2[(a * 4 + b) * 2 + 1] * scalar_w21 * product;
    }
  output[n * width + c] = scalar;
  if (!vectors) return;
  const float vector_w1 = vw1[species * 128 + c];
  const float vector_w20 = vw2[(species * 2) * 128 + c];
  const float vector_w21 = vw2[(species * 2 + 1) * 128 + c];
  for (int m = 0; m < 3; ++m) {
    float value = 0.0f;
    for (int a = 0; a < 4; ++a)
      value += vu1[m * 4 + a] * vector_w1 * x[a];
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) {
        const float product = products[4 * a + b];
        value += vu2[((m * 4 + a) * 4 + b) * 2] * vector_w20 * product;
        value += vu2[((m * 4 + a) * 4 + b) * 2 + 1] * vector_w21 * product;
      }
    output[n * width + 128 + 3 * c + m] = value;
  }
}

__global__ void add_kernel(float *left, const float *right, std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) left[i] += right[i];
}

__global__ void split_condition_kernel(
    const float *sized, const std::uint64_t *batch, const float *graph,
    const float *weight, const float *bias, float *scalar,
    float *equivariant, std::size_t nodes) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nodes * kChannels) return;
  const std::size_t n = i / kChannels, out = i % kChannels;
  float sum = bias[out];
  for (std::size_t in = 0; in < kChannels; ++in)
    sum += sized[n * 512 + in] * weight[in * kChannels + out];
  scalar[i] = sum + graph[batch[n]] * weight[kChannels * kChannels + out];
  for (int m = 0; m < 3; ++m)
    equivariant[n * 384 + 3 * out + m] =
        sized[n * 512 + 128 + 3 * out + m];
}

static int blocks(std::size_t count) {
  return static_cast<int>((count + kBlock - 1) / kBlock);
}

// Result of checking the generated topology against the captured one.
// `remap` sends a generated edge position to its captured position;
// `canonical_hash` is `mace_nlist::topology_hash` over the generated edges
// reordered into capture order. The hash is order-dependent, and capture order
// is the one ordering every backend shares, so it is the only ordering whose
// value is comparable across backends.
struct TopologyReport {
  std::vector<std::size_t> remap;
  std::uint64_t canonical_hash = 0;
};

struct Validation {
  const Fixture &fixture;
  std::vector<std::size_t> edge_remap;
  double atol = 2.0e-4;
  double rtol = 2.0e-3;
  bool pass = true;
  std::size_t checkpoints = 0;
  double max_abs = 0.0, max_rel = 0.0;

  void check(const std::string &name, const float *device, std::size_t count) {
    std::vector<float> fp32_value(count);
    HIP_CHECK(hipMemcpy(fp32_value.data(), device, count * sizeof(float),
                        hipMemcpyDeviceToHost));
    const std::vector<double> value(fp32_value.begin(), fp32_value.end());
    const Tensor &stored = fixture.at("expected_" + name);
    Tensor reordered;
    const bool edge_rows =
        name == "edge_attributes" || name == "edge_features" ||
        (name.size() > 3 && name.compare(name.size() - 3, 3, "_fc") == 0) ||
        (name.size() > 3 && name.compare(name.size() - 3, 3, "_tp") == 0);
    const Tensor *expected = &stored;
    if (edge_rows) {
      const std::size_t width = count / edge_remap.size();
      reordered.shape = {edge_remap.size(), width};
      reordered.f64.resize(count);
      for (std::size_t e = 0; e < edge_remap.size(); ++e)
        std::copy_n(stored.f64.data() + edge_remap[e] * width, width,
                    reordered.f64.data() + e * width);
      expected = &reordered;
    }
    const Error error = compare(value, *expected, atol, rtol);
    pass = pass && error.mismatches == 0;
    max_abs = std::max(max_abs, error.max_abs);
    max_rel = std::max(max_rel, error.max_rel);
    ++checkpoints;
    if (error.mismatches)
      std::cout << "Checkpoint " << name << ": FAIL (mismatches="
                << error.mismatches << ", max_abs=" << std::scientific
                << error.max_abs << ")\n";
  }
};

static std::vector<double> cell_inverses(const Tensor &cells) {
  std::vector<double> result(cells.f64.size());
  for (std::size_t g = 0; g < cells.shape[0]; ++g) {
    const double *a = cells.f64.data() + 9 * g;
    double *iv = result.data() + 9 * g;
    const double det =
        a[0] * (a[4] * a[8] - a[5] * a[7]) -
        a[1] * (a[3] * a[8] - a[5] * a[6]) +
        a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (!std::isfinite(det) || std::abs(det) < 1.0e-14)
      throw std::runtime_error("cell is singular or numerically degenerate");
    const double q = 1.0 / det;
    iv[0] = (a[4] * a[8] - a[5] * a[7]) * q;
    iv[1] = (a[2] * a[7] - a[1] * a[8]) * q;
    iv[2] = (a[1] * a[5] - a[2] * a[4]) * q;
    iv[3] = (a[5] * a[6] - a[3] * a[8]) * q;
    iv[4] = (a[0] * a[8] - a[2] * a[6]) * q;
    iv[5] = (a[2] * a[3] - a[0] * a[5]) * q;
    iv[6] = (a[3] * a[7] - a[4] * a[6]) * q;
    iv[7] = (a[1] * a[6] - a[0] * a[7]) * q;
    iv[8] = (a[0] * a[4] - a[1] * a[3]) * q;
  }
  return result;
}

struct Pipeline {
  const Fixture &host;
  DeviceFixture device;
  rocblas_handle rocblas = nullptr;
  std::size_t nodes, edges;
  Buffer node_attrs, embedded, edge_attrs, edge_features, scalar, equivariant;
  Buffer input, up, down, skip, fc_a, fc_b, messages, aggregate;
  Buffer post, reshaped, sym, product, sized;
  Buffer generated_shifts;
  TypedBuffer<double> inverses, generated_shifts_f64;
  TypedBuffer<Candidate> candidates;
  TypedBuffer<int> neighbor_counts;
  TypedBuffer<std::uint8_t> node_species;
  TypedBuffer<std::uint64_t> neighbor_offsets, generated_edges;
  TypedBuffer<std::uint64_t> scan_totals;
  std::vector<std::size_t> scan_lengths, scan_bases;

  explicit Pipeline(const Fixture &fixture, rocblas_handle blas)
      : host(fixture), device(fixture), rocblas(blas),
        nodes(fixture.at("positions").shape[0]),
        edges(fixture.at("edge_index").shape[0]) {
    if (fixture.version < 2)
      throw std::runtime_error("HIP combined pipeline requires MACEPIPE1 version 2");
    node_attrs.ensure(nodes * 118); embedded.ensure(nodes * 128);
    edge_attrs.ensure(edges * 5); edge_features.ensure(edges * 6);
    scalar.ensure(nodes * 128); equivariant.ensure(nodes * 384);
    input.ensure(nodes * 512); up.ensure(nodes * 512); down.ensure(nodes * 128);
    skip.ensure(nodes * 512); fc_a.ensure(edges * 768); fc_b.ensure(edges * 768);
    messages.ensure(edges * 1024);
    aggregate.ensure(nodes * 1024); post.ensure(nodes * 512);
    reshaped.ensure(nodes * 512); sym.ensure(nodes * 512);
    product.ensure(nodes * 512); sized.ensure(nodes * 512);
    inverses.ensure(fixture.at("cells").f64.size());
    const std::vector<double> inverse_values = cell_inverses(fixture.at("cells"));
    HIP_CHECK(hipMemcpy(inverses.data, inverse_values.data(),
                        inverse_values.size() * sizeof(double),
                        hipMemcpyHostToDevice));
    candidates.ensure(nodes * kMaxNeighbors);
    neighbor_counts.ensure(nodes);
    neighbor_offsets.ensure(nodes + 1);
    std::size_t scan_length = nodes + 1, scan_total = 0;
    while (scan_length > 1) {
      const std::size_t level_blocks =
          (scan_length + kScanBlock - 1) / kScanBlock;
      scan_lengths.push_back(scan_length);
      scan_bases.push_back(scan_total);
      scan_total += level_blocks;
      scan_length = level_blocks;
    }
    scan_totals.ensure(scan_total ? scan_total : 1);
    generated_edges.ensure(edges * 2);
    generated_shifts.ensure(edges * 3);
    generated_shifts_f64.ensure(edges * 3);
    node_species.ensure(nodes);
    std::vector<std::uint8_t> species(nodes);
    const Tensor &raw_x = fixture.at("raw_x");
    for (std::size_t n = 0; n < nodes; ++n) {
      const long value = std::lround(raw_x.f64[n]);
      if (value < 1 || value > static_cast<long>(kSpecies))
        throw std::runtime_error("atomic number outside [1,118]");
      species[n] = static_cast<std::uint8_t>(value - 1);
    }
    HIP_CHECK(hipMemcpy(node_species.data, species.data(), species.size(),
                        hipMemcpyHostToDevice));
  }

  void prefix_scan() {
    const std::size_t levels = scan_lengths.size();
    if (!levels) {
      HIP_CHECK(hipMemset(neighbor_offsets.data, 0, sizeof(std::uint64_t)));
      return;
    }
    for (std::size_t level = 0; level < levels; ++level) {
      const std::size_t length = scan_lengths[level];
      const int grid = static_cast<int>((length + kScanBlock - 1) / kScanBlock);
      std::uint64_t *totals = scan_totals.data + scan_bases[level];
      if (level == 0)
        scan_counts_kernel<<<grid, kScanBlock>>>(
            neighbor_counts.data, nodes, neighbor_offsets.data, totals, length);
      else
        scan_values_kernel<<<grid, kScanBlock>>>(
            scan_totals.data + scan_bases[level - 1], totals, length);
    }
    // The deepest level holds a single block total, whose exclusive prefix is
    // zero, so propagation starts one level below it.
    for (std::size_t level = levels - 1; level-- > 0;) {
      const std::size_t length = scan_lengths[level];
      const int grid = static_cast<int>((length + kScanBlock - 1) / kScanBlock);
      std::uint64_t *values =
          level == 0 ? neighbor_offsets.data
                     : scan_totals.data + scan_bases[level - 1];
      scan_offset_kernel<<<grid, kScanBlock>>>(
          values, scan_totals.data + scan_bases[level], length);
    }
  }

  void reconstruct_neighbors() {
    nlist_kernel<<<static_cast<int>(nodes), kNlistBlock>>>(
        nodes, device.at("graph_ptr").u64, device.at("node_batch").u64,
        device.at("positions").f64, device.at("cells").f64, inverses.data,
        device.at("pbc").u64, candidates.data, neighbor_counts.data);
    prefix_scan();
    flatten_kernel<<<blocks(nodes), kBlock>>>(
        nodes, device.at("node_batch").u64, device.at("cells").f64,
        neighbor_counts.data, neighbor_offsets.data, candidates.data,
        generated_edges.data, generated_shifts_f64.data);
    convert_shifts_kernel<<<blocks(edges * 3), kBlock>>>(
        generated_shifts_f64.data, generated_shifts.data, edges * 3);
    HIP_CHECK(hipGetLastError());
  }

  // Rebuilds the generated edge list as (src, dst, sx, sy, sz) records in the
  // order `flatten_kernel` stores them. The flattened device arrays only keep
  // the physical shift vectors, so the integer images are read back from the
  // candidate slots. Validation-time only: never called from a timed region.
  std::vector<mace_nlist::Edge> stored_topology() {
    std::vector<Candidate> slots(nodes * kMaxNeighbors);
    std::vector<int> counts(nodes);
    HIP_CHECK(hipMemcpy(slots.data(), candidates.data,
                        slots.size() * sizeof(Candidate),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(counts.data(), neighbor_counts.data,
                        counts.size() * sizeof(int), hipMemcpyDeviceToHost));
    std::vector<mace_nlist::Edge> result;
    result.reserve(edges);
    for (std::size_t dst = 0; dst < nodes; ++dst)
      for (int j = 0; j < counts[dst]; ++j) {
        const Candidate &c = slots[dst * kMaxNeighbors + j];
        // Only the integer topology fields feed the hash; the displacement
        // components are left at zero rather than recomputed here.
        result.push_back({c.r2, 0.0, 0.0, 0.0, c.src, static_cast<int>(dst),
                          c.sx, c.sy, c.sz});
      }
    if (result.size() != edges)
      throw std::runtime_error("candidate slot counts disagree with edge total");
    return result;
  }

  TopologyReport validate_topology() {
    std::uint64_t total = 0;
    HIP_CHECK(hipMemcpy(&total, neighbor_offsets.data + nodes,
                        sizeof(total), hipMemcpyDeviceToHost));
    if (total != edges)
      throw std::runtime_error("generated edge count mismatch: got " +
                               std::to_string(total) + ", expected " +
                               std::to_string(edges));
    std::vector<std::uint64_t> got_edges(2 * edges);
    std::vector<double> got_shifts(3 * edges);
    HIP_CHECK(hipMemcpy(got_edges.data(), generated_edges.data,
                        got_edges.size() * sizeof(std::uint64_t),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(got_shifts.data(), generated_shifts_f64.data,
                        got_shifts.size() * sizeof(double),
                        hipMemcpyDeviceToHost));
    const Tensor &wanted_edges = host.at("edge_index");
    const Tensor &wanted_shifts = host.at("edge_shifts");
    std::vector<std::size_t> remap(edges, edges);
    std::vector<bool> used(edges, false);
    for (std::size_t generated = 0; generated < edges; ++generated) {
      for (std::size_t captured = 0; captured < edges; ++captured) {
        if (used[captured] ||
            got_edges[2 * generated] != wanted_edges.u64[2 * captured] ||
            got_edges[2 * generated + 1] != wanted_edges.u64[2 * captured + 1])
          continue;
        bool same = true;
        for (int m = 0; m < 3; ++m)
          same = same &&
              std::abs(got_shifts[3 * generated + m] -
                       wanted_shifts.f64[3 * captured + m]) <= 1.0e-12;
        if (same) {
          remap[generated] = captured;
          used[captured] = true;
          break;
        }
      }
      if (remap[generated] == edges) {
        double closest = std::numeric_limits<double>::infinity();
        for (std::size_t captured = 0; captured < edges; ++captured) {
          if (got_edges[2 * generated] != wanted_edges.u64[2 * captured] ||
              got_edges[2 * generated + 1] != wanted_edges.u64[2 * captured + 1])
            continue;
          double difference = 0.0;
          for (int m = 0; m < 3; ++m)
            difference = std::max(
                difference,
                std::abs(static_cast<double>(got_shifts[3 * generated + m]) -
                         wanted_shifts.f64[3 * captured + m]));
          closest = std::min(closest, difference);
        }
        throw std::runtime_error(
            "generated topology/shift validation failed at edge " +
            std::to_string(generated) + " (closest shift difference " +
            std::to_string(closest) + ")");
      }
    }
    const std::vector<mace_nlist::Edge> stored = stored_topology();
    std::vector<mace_nlist::Edge> canonical(edges);
    for (std::size_t generated = 0; generated < edges; ++generated)
      canonical[remap[generated]] = stored[generated];
    TopologyReport report;
    report.canonical_hash = mace_nlist::topology_hash(canonical);
    report.remap = std::move(remap);
    return report;
  }

  void matmul(const float *x, const std::string &matrix_name, float *y,
              std::size_t rows) {
    const Tensor &m = host.at(matrix_name);
    const std::size_t in_width = m.shape[0], out_width = m.shape[1];
    const float alpha = 1.0f, beta = 0.0f;
    ROCBLAS_CHECK(rocblas_sgemm(
        rocblas, rocblas_operation_none, rocblas_operation_none,
        gemm_index<int>(out_width, "N"), gemm_index<int>(rows, "M"),
        gemm_index<int>(in_width, "K"), &alpha,
        device.at(matrix_name).f32, gemm_index<int>(out_width, "ldA"), x,
        gemm_index<int>(in_width, "ldB"), &beta, y,
        gemm_index<int>(out_width, "ldC")));
  }

  void checkpoint(Validation *v, const std::string &name, const Buffer &b,
                  std::size_t count) {
    HIP_CHECK(hipGetLastError());
    if (v) v->check(name, b.data, count);
  }

  void run(Validation *validation = nullptr) {
    // Encode atomic numbers, using the explicit one-hot path for checkpoint
    // validation and the equivalent direct embedding lookup for timed runs.
    if (validation) {
      one_hot_kernel<<<blocks(nodes * 118), kBlock>>>(
          device.at("raw_x").f32, node_attrs.data, nodes);
      checkpoint(validation, "node_attributes", node_attrs, nodes * 118);
      matmul(node_attrs.data, "node_embedding", embedded.data, nodes);
    } else {
      embedding_gather_kernel<<<blocks(nodes * 128), kBlock>>>(
          node_species.data, device.at("node_embedding").f32,
          embedded.data, nodes);
    }
    checkpoint(validation, "node_embedding", embedded, nodes * 128);

    // Build distance, direction, and radial-basis features for every edge.
    edge_kernel<<<blocks(edges), kBlock>>>(
        device.at("positions").f32, generated_edges.data,
        generated_shifts.data, device.at("bessel_weights").f32,
        device.at("radial_constants").f32,
        edge_attrs.data, edge_features.data, edges);
    checkpoint(validation, "edge_attributes", edge_attrs, edges * 5);
    checkpoint(validation, "edge_features", edge_features, edges * 6);

    // Inject each graph's conditioning scalar into its initial node features.
    condition_kernel<<<blocks(nodes * 128), kBlock>>>(
        embedded.data, device.at("node_batch").u64, device.at("graph_attr").f32,
        device.at("condition_weight").f32, device.at("condition_bias").f32,
        scalar.data, nodes);

    for (int layer = 0; layer < 4; ++layer) {
      const std::string prefix = "layer" + std::to_string(layer) + "_";

      // Pack scalar and equivariant channels, then form the three learned
      // projections used by message passing and the residual connection.
      combine_kernel<<<blocks(nodes * (layer ? 512 : 128)), kBlock>>>(
          scalar.data, equivariant.data, input.data, nodes, layer != 0);
      checkpoint(validation, prefix + "input", input, nodes * (layer ? 512 : 128));
      matmul(input.data, prefix + "linear_up", up.data, nodes);
      matmul(input.data, prefix + "linear_down", down.data, nodes);
      matmul(input.data, prefix + "linear_skip", skip.data, nodes);
      checkpoint(validation, prefix + "up", up,
                 nodes * host.at(prefix + "linear_up").shape[1]);
      checkpoint(validation, prefix + "down", down, nodes * 128);
      const std::size_t skip_width = host.at(prefix + "linear_skip").shape[1];
      checkpoint(validation, prefix + "skip", skip, nodes * skip_width);

      // Concatenate radial and endpoint features, then evaluate the edge MLP
      // that produces tensor-product weights.
      fc_input_kernel<<<blocks(edges * 262), kBlock>>>(
          edge_features.data, down.data, generated_edges.data,
          fc_a.data, edges);
      float *current = fc_a.data;
      float *next = fc_b.data;
      std::size_t current_width = 262;
      for (int stage = 0; stage < 4; ++stage) {
        const std::string name = prefix + "fc" + std::to_string(stage);
        const std::size_t next_width = host.at(name).shape[1];
        matmul(current, name, next, edges);
        if (stage < 3)
          silu_kernel<<<blocks(edges * next_width), kBlock>>>(
              next, edges * next_width);
        std::swap(current, next);
        current_width = next_width;
      }
      HIP_CHECK(hipGetLastError());
      if (validation)
        validation->check(prefix + "fc", current, edges * current_width);
      const std::size_t tp_width = layer == 0 ? 512 : 1024;

      // Validation materializes per-edge tensor products before aggregation;
      // timed runs fuse both operations to avoid the large message buffer.
      if (validation) {
        tensor_product_kernel<<<blocks(edges * tp_width), kBlock>>>(
            layer, generated_edges.data, up.data, edge_attrs.data,
            current, messages.data, edges);
        checkpoint(validation, prefix + "tp", messages, edges * tp_width);
        aggregate_kernel<<<blocks(nodes * tp_width), kBlock>>>(
            neighbor_offsets.data, messages.data, aggregate.data, nodes, tp_width);
      } else {
        tensor_product_aggregate_kernel<<<blocks(nodes * kChannels), kBlock>>>(
            layer, generated_edges.data, neighbor_offsets.data, up.data,
            edge_attrs.data, current, aggregate.data, nodes);
      }
      checkpoint(validation, prefix + "aggregate", aggregate, nodes * tp_width);

      // Project aggregated messages back to node channels.
      matmul(aggregate.data, prefix + "linear_post", post.data, nodes);
      checkpoint(validation, prefix + "post", post, nodes * 512);

      // Normalize and reinterpret the channels as four irrep components.
      // The timed path fuses these two elementwise transformations.
      if (validation) {
        scale_kernel<<<blocks(nodes * 512), kBlock>>>(
            post.data, nodes * 512, device.at("radial_constants").f32);
        reshape_kernel<<<blocks(nodes * 512), kBlock>>>(
            post.data, reshaped.data, nodes * 512);
      } else {
        const float divisor =
            static_cast<float>(host.at("radial_constants").f64[3]);
        scale_reshape_kernel<<<blocks(nodes * 512), kBlock>>>(
            post.data, reshaped.data, nodes * 512, divisor);
      }
      checkpoint(validation, prefix + "reshape", reshaped, nodes * 512);
      const bool vectors = layer < 3;

      // Apply the species-dependent symmetric contraction.
      sym_kernel<<<blocks(nodes * kChannels), kBlock>>>(
          layer, reshaped.data, node_species.data,
          device.at(prefix + "scalar_w1").f32, device.at(prefix + "scalar_w2").f32,
          device.at(prefix + "scalar_u1").f32, device.at(prefix + "scalar_u2").f32,
          vectors ? device.at(prefix + "vector_w1").f32 : nullptr,
          vectors ? device.at(prefix + "vector_w2").f32 : nullptr,
          vectors ? device.at(prefix + "vector_u1").f32 : nullptr,
          vectors ? device.at(prefix + "vector_u2").f32 : nullptr,
          sym.data, nodes);
      const std::size_t sym_width = vectors ? 512 : 128;
      checkpoint(validation, prefix + "sym", sym, nodes * sym_width);

      // Transform the contracted features, add the residual projection, and
      // size the result for the next layer (or final output).
      matmul(sym.data, prefix + "linear_product", product.data, nodes);
      const std::size_t product_width = host.at(prefix + "linear_product").shape[1];
      checkpoint(validation, prefix + "prod_linear", product,
                 nodes * product_width);
      add_kernel<<<blocks(nodes * product_width), kBlock>>>(
          product.data, skip.data, nodes * product_width);
      checkpoint(validation, prefix + "product", product, nodes * product_width);
      matmul(product.data, prefix + "linear_sizing", sized.data, nodes);
      const std::size_t sized_width = host.at(prefix + "linear_sizing").shape[1];
      checkpoint(validation, prefix + "sizing", sized, nodes * sized_width);
      if (layer < 3) {
        // Split the sized irreps and reapply graph conditioning for the next
        // interaction layer.
        split_condition_kernel<<<blocks(nodes * kChannels), kBlock>>>(
            sized.data, device.at("node_batch").u64,
            device.at("graph_attr").f32, device.at("condition_weight").f32,
            device.at("condition_bias").f32, scalar.data, equivariant.data,
            nodes);
      }
    }
    HIP_CHECK(hipGetLastError());
  }
};

static int parse_count(const char *text, const char *name, bool positive) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end || value < (positive ? 1 : 0) ||
      value > std::numeric_limits<int>::max())
    throw std::runtime_error(std::string("invalid ") + name);
  return static_cast<int>(value);
}

template <class Function>
static double measure(Function function, int repeat) {
  HIP_CHECK(hipDeviceSynchronize());
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) function();
  HIP_CHECK(hipDeviceSynchronize());
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count() /
         repeat;
}

int main(int argc, char **argv) {
  try {
    std::string fixture_path;
    int warmup = 1, repeat = 3;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      auto value = [&]() {
        if (++i == argc) throw std::runtime_error("missing option value");
        return argv[i];
      };
      if (argument == "--fixture") fixture_path = value();
      else if (argument == "--warmup")
        warmup = parse_count(value(), "--warmup", false);
      else if (argument == "--repeat")
        repeat = parse_count(value(), "--repeat", true);
      else if (argument == "--help" || argument == "-h") {
        std::cout << "Usage: " << argv[0]
                  << " --fixture FILE [--warmup N] [--repeat N]\n";
        return 0;
      } else throw std::runtime_error("unknown option: " + argument);
    }
    if (fixture_path.empty()) throw std::runtime_error("--fixture is required");
    int device_id = 0;
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&properties, device_id));
    const Fixture fixture = read_fixture(fixture_path);
    BlasContext blas;
    Pipeline pipeline(fixture, blas.handle);
    std::cout << "MACE combined convolution pipeline\n"
              << "Backend: HIP (" << properties.name << ", "
              << properties.gcnArchName << ")\n"
              << "Precision: FP32 (MACE), FP64 (neighbor geometry)\nNodes: "
              << pipeline.nodes
              << "\nEdges: " << pipeline.edges
              << "\nLayers: 4\nFinal shape: [" << pipeline.nodes << ",128]\n";
    pipeline.reconstruct_neighbors();
    const TopologyReport topology = pipeline.validate_topology();
    std::size_t reordered = 0;
    for (std::size_t e = 0; e < topology.remap.size(); ++e)
      reordered += topology.remap[e] != e;
    std::cout << "Neighbor topology validation: PASS (" << pipeline.edges
              << " edges, " << reordered
              << " canonical positions differ from capture order, hash=0x"
              << std::hex << topology.canonical_hash << std::dec << ")\n";
    Validation validation{fixture, topology.remap};
    pipeline.run(&validation);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "Validation: " << (validation.pass ? "PASS" : "FAIL")
              << " (" << validation.checkpoints << " checkpoints, max_abs="
              << std::scientific << validation.max_abs << ", max_rel="
              << validation.max_rel << ")\n";
    if (!validation.pass) return 2;

    // The checkpoints above only cover the explicit path; run the fused
    // kernels the timed loops use and validate their final output too.
    pipeline.run();
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> fast_fp32(pipeline.nodes * kChannels);
    HIP_CHECK(hipMemcpy(fast_fp32.data(), pipeline.sized.data,
                        fast_fp32.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    const std::vector<double> fast_output(fast_fp32.begin(), fast_fp32.end());
    const Error fast_error =
        compare(fast_output, fixture.at("expected_layer3_sizing"),
                validation.atol, validation.rtol);
    std::cout << "Timed fast-path validation: "
              << (fast_error.mismatches == 0 ? "PASS" : "FAIL")
              << " (max_abs=" << std::scientific << fast_error.max_abs
              << ", max_rel=" << fast_error.max_rel << ")\n";
    if (fast_error.mismatches) return 2;

    // Correctness above is always established on the captured fixture, the only
    // input with reference tensors. The timed region then runs a synthetic batch
    // at the published SC26 dimensions.
    const production::Workload workload = production::generate(fixture);
    Pipeline scaled(workload.fixture, blas.handle);
    Pipeline *timed = &scaled;
    std::cout << "Timed workload: synthetic production batch ("
              << workload.graphs << " graphs, " << workload.nodes
              << " nodes, " << workload.edges << " edges";
    if (workload.edges != workload.requested_edges)
      std::cout << ", requested " << workload.requested_edges;
    std::cout << ")\n";
    timed->reconstruct_neighbors();
    const std::uint64_t built =
        mace_nlist::topology_hash(timed->stored_topology());
    if (built != workload.topology_hash)
      throw std::runtime_error(
          "production topology disagrees with the host reference");
    std::cout << "Production topology check: PASS (hash=0x" << std::hex
              << built << std::dec << ")\n";

    for (int i = 0; i < warmup; ++i) {
      timed->reconstruct_neighbors();
      timed->run();
    }
    HIP_CHECK(hipDeviceSynchronize());
    const double nlist_ms = measure(
        [&] { timed->reconstruct_neighbors(); }, repeat);
    const double convolution_ms = measure(
        [&] { timed->run(); }, repeat);
    const double total_ms = nlist_ms + convolution_ms;
    std::cout << std::fixed << std::setprecision(6)
              << "Average neighbor reconstruction time: " << nlist_ms << " ms\n"
              << "Average convolution-only time: " << convolution_ms << " ms\n"
              << "Average total time: " << total_ms << " ms\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
