#ifndef HECBENCH_MACE_PIPELINE_HPP
#define HECBENCH_MACE_PIPELINE_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "mace_nlist.hpp"

namespace mace_pipeline {

constexpr std::size_t kChannels = 128;
constexpr std::size_t kSpecies = 118;
constexpr std::size_t kLayers = 4;
constexpr double kSiluScale = 1.6791767923989418;

template <typename Index>
inline Index gemm_index(std::size_t value, const char *name) {
  if (value > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
    throw std::runtime_error(std::string("GEMM ") + name +
                             " exceeds the BLAS index type");
  return static_cast<Index>(value);
}

struct Tensor {
  std::vector<std::uint64_t> shape;
  std::vector<double> f64;
  std::vector<std::uint64_t> u64;
  std::size_t size() const { return f64.empty() ? u64.size() : f64.size(); }
};

struct Fixture {
  std::uint32_t version = 0;
  std::map<std::string, Tensor> tensors;
  const Tensor &at(const std::string &name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("missing fixture tensor: " + name);
    return it->second;
  }
};

template <typename T>
inline T read_scalar(std::ifstream &in, const char *field) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!in) throw std::runtime_error(std::string("truncated fixture field: ") + field);
  return value;
}

inline std::size_t checked_count(const std::vector<std::uint64_t> &shape) {
  std::size_t count = 1;
  for (std::uint64_t dim : shape) {
    if (dim > std::numeric_limits<std::size_t>::max() ||
        (dim && count > std::numeric_limits<std::size_t>::max() / dim))
      throw std::runtime_error("fixture tensor dimensions overflow");
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

inline Fixture read_fixture(const std::string &path) {
  const std::uint16_t endian = 1;
  if (*reinterpret_cast<const unsigned char *>(&endian) != 1)
    throw std::runtime_error("MACEPIPE1 requires a little-endian host");
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open fixture: " + path);
  char magic[9]{};
  in.read(magic, 9);
  if (!in || std::memcmp(magic, "MACEPIPE1", 9))
    throw std::runtime_error("invalid MACEPIPE1 fixture");
  const std::uint32_t version = read_scalar<std::uint32_t>(in, "version");
  const std::uint32_t tensors = read_scalar<std::uint32_t>(in, "tensor count");
  if (version != 1 && version != 2)
    throw std::runtime_error("unsupported MACEPIPE1 version");
  Fixture fixture;
  fixture.version = version;
  for (std::uint32_t index = 0; index < tensors; ++index) {
    const std::uint16_t name_size = read_scalar<std::uint16_t>(in, "name size");
    const std::uint8_t rank = read_scalar<std::uint8_t>(in, "rank");
    const std::uint8_t type = read_scalar<std::uint8_t>(in, "type");
    if (!name_size || rank > 8 || type > 1) throw std::runtime_error("invalid tensor header");
    std::string name(name_size, '\0');
    in.read(name.data(), name.size());
    Tensor tensor;
    tensor.shape.resize(rank);
    for (auto &dim : tensor.shape) dim = read_scalar<std::uint64_t>(in, "dimension");
    const std::size_t count = checked_count(tensor.shape);
    if (type == 0) {
      tensor.f64.resize(count);
      in.read(reinterpret_cast<char *>(tensor.f64.data()),
              static_cast<std::streamsize>(count * sizeof(double)));
    } else {
      tensor.u64.resize(count);
      in.read(reinterpret_cast<char *>(tensor.u64.data()),
              static_cast<std::streamsize>(count * sizeof(std::uint64_t)));
    }
    if (!in) throw std::runtime_error("truncated tensor: " + name);
    if (!fixture.tensors.emplace(name, std::move(tensor)).second)
      throw std::runtime_error("duplicate tensor: " + name);
  }
  if (in.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("trailing fixture bytes");
  if (version >= 2) {
    const Tensor &positions = fixture.at("positions");
    const Tensor &ptr = fixture.at("graph_ptr");
    const Tensor &cells = fixture.at("cells");
    const Tensor &pbc = fixture.at("pbc");
    if (positions.shape.size() != 2 || positions.shape[1] != 3 ||
        ptr.shape.size() != 1 || ptr.u64.size() < 2 ||
        ptr.u64.front() != 0 || ptr.u64.back() != positions.shape[0])
      throw std::runtime_error("invalid MACEPIPE2 graph offsets");
    const std::size_t graphs = ptr.u64.size() - 1;
    if (cells.shape != std::vector<std::uint64_t>{graphs, 3, 3} ||
        pbc.shape != std::vector<std::uint64_t>{graphs, 3})
      throw std::runtime_error("invalid MACEPIPE2 cell/PBC dimensions");
    for (std::size_t i = 1; i < ptr.u64.size(); ++i)
      if (ptr.u64[i] < ptr.u64[i - 1])
        throw std::runtime_error("MACEPIPE2 graph offsets are not monotonic");
    for (std::uint64_t periodic : pbc.u64)
      if (periodic > 1) throw std::runtime_error("invalid MACEPIPE2 PBC flag");
  }
  return fixture;
}

inline std::string lname(std::size_t layer, const std::string &suffix) {
  return "layer" + std::to_string(layer) + "_" + suffix;
}

struct Error {
  std::size_t mismatches = 0;
  double max_abs = 0.0;
  double max_rel = 0.0;
};

inline Error compare(const std::vector<double> &got, const Tensor &expected,
                     double atol, double rtol) {
  if (got.size() != expected.f64.size())
    throw std::runtime_error("checkpoint size mismatch");
  Error error;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double absolute = std::abs(got[i] - expected.f64[i]);
    const double relative = absolute / std::max(std::abs(expected.f64[i]), 1.0e-300);
    error.max_abs = std::max(error.max_abs, absolute);
    error.max_rel = std::max(error.max_rel, relative);
    if (!std::isfinite(got[i]) ||
        absolute > atol + rtol * std::abs(expected.f64[i]))
      ++error.mismatches;
  }
  return error;
}

using Checkpoint = void (*)(void *, const std::string &, const std::vector<double> &);

inline void matmul(const std::vector<double> &input, std::size_t rows,
                   const Tensor &matrix, std::vector<double> &output) {
  if (matrix.shape.size() != 2 || input.size() != rows * matrix.shape[0])
    throw std::runtime_error("invalid matrix multiplication dimensions");
  const std::size_t in_width = matrix.shape[0];
  const std::size_t out_width = matrix.shape[1];
  output.assign(rows * out_width, 0.0);
#pragma omp parallel for schedule(static)
  for (std::int64_t signed_row = 0; signed_row < static_cast<std::int64_t>(rows);
       ++signed_row) {
    const std::size_t row = static_cast<std::size_t>(signed_row);
    for (std::size_t out = 0; out < out_width; ++out) {
      double sum = 0.0;
      for (std::size_t in = 0; in < in_width; ++in)
        sum += input[row * in_width + in] *
               matrix.f64[in * out_width + out];
      output[row * out_width + out] = sum;
    }
  }
}

inline void edge_preparation(const Fixture &f, std::vector<double> &attributes,
                             std::vector<double> &features) {
  const Tensor &positions = f.at("positions");
  const Tensor &edges = f.at("edge_index");
  const Tensor &shifts = f.at("edge_shifts");
  const Tensor &raw = f.at("edge_attr_raw");
  const Tensor &bessel = f.at("bessel_weights");
  const Tensor &constants = f.at("radial_constants");
  const std::size_t edge_count = edges.shape[0];
  attributes.resize(edge_count * 5);
  features.resize(edge_count * 6);
  const double prefactor = constants.f64[0];
  const double r_max = constants.f64[1];
  const double p = constants.f64[2];
  const double root3 = std::sqrt(3.0);
#pragma omp parallel for schedule(static)
  for (std::int64_t se = 0; se < static_cast<std::int64_t>(edge_count); ++se) {
    const std::size_t e = static_cast<std::size_t>(se);
    const std::size_t sender = edges.u64[2 * e];
    const std::size_t receiver = edges.u64[2 * e + 1];
    double vector[3];
    double r2 = 0.0;
    for (std::size_t m = 0; m < 3; ++m) {
      vector[m] = positions.f64[3 * receiver + m] -
                  positions.f64[3 * sender + m] + shifts.f64[3 * e + m];
      r2 += vector[m] * vector[m];
    }
    const double r = std::sqrt(r2);
    attributes[5 * e] = raw.f64[e];
    attributes[5 * e + 1] = 1.0;
    for (std::size_t m = 0; m < 3; ++m)
      attributes[5 * e + 2 + m] = root3 * vector[m] / r;
    const double q = r / r_max;
    const double cutoff = r < r_max
        ? 1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * std::pow(q, p)
              + p * (p + 2.0) * std::pow(q, p + 1.0)
              - (p * (p + 1.0) / 2.0) * std::pow(q, p + 2.0)
        : 0.0;
    for (std::size_t k = 0; k < 6; ++k)
      features[6 * e + k] =
          prefactor * std::sin(bessel.f64[k] * r) / r * cutoff;
  }
}

inline void condition(const Fixture &f, const std::vector<double> &scalar,
                      std::vector<double> &result) {
  const Tensor &batch = f.at("node_batch");
  const Tensor &graph = f.at("graph_attr");
  const Tensor &weight = f.at("condition_weight");
  const Tensor &bias = f.at("condition_bias");
  const std::size_t nodes = batch.u64.size();
  result.resize(nodes * kChannels);
#pragma omp parallel for schedule(static)
  for (std::int64_t sn = 0; sn < static_cast<std::int64_t>(nodes); ++sn) {
    const std::size_t n = static_cast<std::size_t>(sn);
    for (std::size_t out = 0; out < kChannels; ++out) {
      double sum = bias.f64[out];
      for (std::size_t in = 0; in < kChannels; ++in)
        sum += scalar[n * kChannels + in] * weight.f64[in * kChannels + out];
      sum += graph.f64[batch.u64[n]] * weight.f64[kChannels * kChannels + out];
      result[n * kChannels + out] = sum;
    }
  }
}

inline void fc_weights(const Fixture &f, std::size_t layer,
                       const std::vector<double> &edge_features,
                       const std::vector<double> &down,
                       std::vector<double> &weights) {
  const Tensor &edges = f.at("edge_index");
  const std::size_t edge_count = edges.shape[0];
  std::vector<double> current(edge_count * 262);
#pragma omp parallel for schedule(static)
  for (std::int64_t se = 0; se < static_cast<std::int64_t>(edge_count); ++se) {
    const std::size_t e = static_cast<std::size_t>(se);
    const std::size_t sender = edges.u64[2 * e];
    const std::size_t receiver = edges.u64[2 * e + 1];
    std::copy_n(edge_features.data() + 6 * e, 6, current.data() + 262 * e);
    std::copy_n(down.data() + kChannels * sender, kChannels,
                current.data() + 262 * e + 6);
    std::copy_n(down.data() + kChannels * receiver, kChannels,
                current.data() + 262 * e + 134);
  }
  for (std::size_t stage = 0; stage < 4; ++stage) {
    std::vector<double> next;
    matmul(current, edge_count, f.at(lname(layer, "fc" + std::to_string(stage))), next);
    if (stage < 3) {
#pragma omp parallel for schedule(static)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(next.size()); ++i) {
        const double x = next[static_cast<std::size_t>(i)];
        next[static_cast<std::size_t>(i)] =
            kSiluScale * x / (1.0 + std::exp(-x));
      }
    }
    current.swap(next);
  }
  weights.swap(current);
}

inline void tensor_product(std::size_t layer, const Tensor &edges,
                           const std::vector<double> &up,
                           const std::vector<double> &attrs,
                           const std::vector<double> &weights,
                           std::vector<double> &messages) {
  const std::size_t edge_count = edges.shape[0];
  const std::size_t width = layer == 0 ? 512 : 1024;
  const std::size_t input_width = layer == 0 ? 128 : 512;
  messages.assign(edge_count * width, 0.0);
  const double inv_root2 = 1.0 / std::sqrt(2.0);
  const double inv_root3 = 1.0 / std::sqrt(3.0);
#pragma omp parallel for schedule(static)
  for (std::int64_t se = 0; se < static_cast<std::int64_t>(edge_count); ++se) {
    const std::size_t e = static_cast<std::size_t>(se);
    const std::size_t sender = edges.u64[2 * e];
    const double *x = up.data() + sender * input_width;
    const double *a = attrs.data() + 5 * e;
    const double *w = weights.data() + e * (layer == 0 ? 384 : 768);
    double *y = messages.data() + e * width;
    for (std::size_t c = 0; c < kChannels; ++c) {
      y[c] = x[c] * (w[2 * c] * a[0] + w[2 * c + 1] * a[1]) * inv_root2;
      const std::size_t vector_out = layer == 0 ? 128 + 3 * c : 256 + 3 * c;
      const std::size_t vector_weight = layer == 0 ? 256 + c : 384 + c;
      for (std::size_t m = 0; m < 3; ++m)
        y[vector_out + m] = x[c] * a[2 + m] * w[vector_weight];
      if (layer) {
        double dot = 0.0;
        for (std::size_t m = 0; m < 3; ++m)
          dot += x[128 + 3 * c + m] * a[2 + m];
        y[128 + c] = dot * w[256 + c] * inv_root3;
        for (std::size_t m = 0; m < 3; ++m)
          y[640 + 3 * c + m] = x[128 + 3 * c + m] *
              (w[512 + 2 * c] * a[0] + w[513 + 2 * c] * a[1]) * inv_root2;
      }
    }
  }
}

inline void aggregate(const Tensor &edges, std::size_t nodes,
                      const std::vector<double> &messages, std::size_t width,
                      std::vector<double> &output) {
  const std::size_t edge_count = edges.shape[0];
  std::vector<std::size_t> offsets(nodes + 1, 0);
  for (std::size_t e = 0; e < edge_count; ++e)
    ++offsets[edges.u64[2 * e + 1] + 1];
  for (std::size_t n = 0; n < nodes; ++n) offsets[n + 1] += offsets[n];
  output.assign(nodes * width, 0.0);
#pragma omp parallel for schedule(static)
  for (std::int64_t index = 0;
       index < static_cast<std::int64_t>(nodes * width); ++index) {
    const std::size_t n = static_cast<std::size_t>(index) / width;
    const std::size_t c = static_cast<std::size_t>(index) % width;
    double sum = 0.0;
    for (std::size_t e = offsets[n]; e < offsets[n + 1]; ++e)
      sum += messages[e * width + c];
    output[static_cast<std::size_t>(index)] = sum;
  }
}

inline void reshape(const std::vector<double> &post, std::size_t nodes,
                    std::vector<double> &output) {
  output.resize(nodes * 512);
#pragma omp parallel for schedule(static)
  for (std::int64_t sn = 0; sn < static_cast<std::int64_t>(nodes); ++sn) {
    const std::size_t n = static_cast<std::size_t>(sn);
    for (std::size_t c = 0; c < kChannels; ++c) {
      output[(n * kChannels + c) * 4] = post[n * 512 + c];
      for (std::size_t m = 0; m < 3; ++m)
        output[(n * kChannels + c) * 4 + 1 + m] =
            post[n * 512 + 128 + 3 * c + m];
    }
  }
}

inline void symmetric_contraction(const Fixture &f, std::size_t layer,
                                  const std::vector<double> &input,
                                  std::vector<double> &output) {
  const Tensor &attrs = f.at("expected_node_attributes");
  const std::size_t nodes = attrs.shape[0];
  const bool vectors = layer < 3;
  const std::size_t width = vectors ? 512 : 128;
  output.assign(nodes * width, 0.0);
  const Tensor &sw1 = f.at(lname(layer, "scalar_w1"));
  const Tensor &sw2 = f.at(lname(layer, "scalar_w2"));
  const Tensor &su1 = f.at(lname(layer, "scalar_u1"));
  const Tensor &su2 = f.at(lname(layer, "scalar_u2"));
  const Tensor *vw1 = vectors ? &f.at(lname(layer, "vector_w1")) : nullptr;
  const Tensor *vw2 = vectors ? &f.at(lname(layer, "vector_w2")) : nullptr;
  const Tensor *vu1 = vectors ? &f.at(lname(layer, "vector_u1")) : nullptr;
  const Tensor *vu2 = vectors ? &f.at(lname(layer, "vector_u2")) : nullptr;
#pragma omp parallel for schedule(static)
  for (std::int64_t index = 0;
       index < static_cast<std::int64_t>(nodes * kChannels); ++index) {
    const std::size_t n = static_cast<std::size_t>(index) / kChannels;
    const std::size_t c = static_cast<std::size_t>(index) % kChannels;
    std::size_t species = 0;
    while (species < kSpecies && attrs.f64[n * kSpecies + species] != 1.0) ++species;
    if (species == kSpecies) continue;
    const double *x = input.data() + (n * kChannels + c) * 4;
    double scalar = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
      scalar += su1.f64[i] * sw1.f64[(species * 1) * kChannels + c] * x[i];
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = 0; j < 4; ++j)
        for (std::size_t p = 0; p < 2; ++p)
          scalar += su2.f64[(i * 4 + j) * 2 + p] *
              sw2.f64[(species * 2 + p) * kChannels + c] * x[i] * x[j];
    output[n * width + c] = scalar;
    if (!vectors) continue;
    for (std::size_t m = 0; m < 3; ++m) {
      double vector = 0.0;
      for (std::size_t i = 0; i < 4; ++i)
        vector += vu1->f64[m * 4 + i] *
            vw1->f64[species * kChannels + c] * x[i];
      for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
          for (std::size_t p = 0; p < 2; ++p)
            vector += vu2->f64[((m * 4 + i) * 4 + j) * 2 + p] *
                vw2->f64[(species * 2 + p) * kChannels + c] * x[i] * x[j];
      output[n * width + 128 + 3 * c + m] = vector;
    }
  }
}

inline void run(const Fixture &f, Checkpoint checkpoint, void *context,
                std::vector<double> &final) {
  const std::size_t nodes = f.at("positions").shape[0];
  const std::size_t edges = f.at("edge_index").shape[0];
  std::vector<double> node_attrs(nodes * kSpecies, 0.0);
  const Tensor &raw_x = f.at("raw_x");
  for (std::size_t n = 0; n < nodes; ++n) {
    const long atomic_number = std::lround(raw_x.f64[n]);
    if (atomic_number < 1 || atomic_number > static_cast<long>(kSpecies))
      throw std::runtime_error("atomic number outside [1,118]");
    node_attrs[n * kSpecies + static_cast<std::size_t>(atomic_number - 1)] = 1.0;
  }
  checkpoint(context, "node_attributes", node_attrs);
  std::vector<double> embedded;
  matmul(node_attrs, nodes, f.at("node_embedding"), embedded);
  checkpoint(context, "node_embedding", embedded);
  std::vector<double> edge_attrs, edge_features;
  edge_preparation(f, edge_attrs, edge_features);
  checkpoint(context, "edge_attributes", edge_attrs);
  checkpoint(context, "edge_features", edge_features);
  std::vector<double> scalar;
  condition(f, embedded, scalar);
  std::vector<double> equivariant;
  for (std::size_t layer = 0; layer < kLayers; ++layer) {
    std::vector<double> input;
    input.reserve(scalar.size() + equivariant.size());
    for (std::size_t n = 0; n < nodes; ++n) {
      input.insert(input.end(), scalar.begin() + n * 128, scalar.begin() + (n + 1) * 128);
      if (!equivariant.empty())
        input.insert(input.end(), equivariant.begin() + n * 384,
                     equivariant.begin() + (n + 1) * 384);
    }
    checkpoint(context, lname(layer, "input"), input);
    std::vector<double> up, down, skip;
    matmul(input, nodes, f.at(lname(layer, "linear_up")), up);
    matmul(input, nodes, f.at(lname(layer, "linear_down")), down);
    matmul(input, nodes, f.at(lname(layer, "linear_skip")), skip);
    checkpoint(context, lname(layer, "up"), up);
    checkpoint(context, lname(layer, "down"), down);
    checkpoint(context, lname(layer, "skip"), skip);
    std::vector<double> weights;
    fc_weights(f, layer, edge_features, down, weights);
    checkpoint(context, lname(layer, "fc"), weights);
    std::vector<double> messages;
    tensor_product(layer, f.at("edge_index"), up, edge_attrs, weights, messages);
    checkpoint(context, lname(layer, "tp"), messages);
    std::vector<double> aggregate_values;
    aggregate(f.at("edge_index"), nodes, messages, layer == 0 ? 512 : 1024,
              aggregate_values);
    checkpoint(context, lname(layer, "aggregate"), aggregate_values);
    std::vector<double> post;
    matmul(aggregate_values, nodes, f.at(lname(layer, "linear_post")), post);
    checkpoint(context, lname(layer, "post"), post);
    const double avg = f.at("radial_constants").f64[3];
    for (double &value : post) value /= avg;
    std::vector<double> reshaped;
    reshape(post, nodes, reshaped);
    checkpoint(context, lname(layer, "reshape"), reshaped);
    std::vector<double> sym;
    symmetric_contraction(f, layer, reshaped, sym);
    checkpoint(context, lname(layer, "sym"), sym);
    std::vector<double> product_linear;
    matmul(sym, nodes, f.at(lname(layer, "linear_product")), product_linear);
    checkpoint(context, lname(layer, "prod_linear"), product_linear);
    for (std::size_t i = 0; i < product_linear.size(); ++i)
      product_linear[i] += skip[i];
    checkpoint(context, lname(layer, "product"), product_linear);
    std::vector<double> sized;
    matmul(product_linear, nodes, f.at(lname(layer, "linear_sizing")), sized);
    checkpoint(context, lname(layer, "sizing"), sized);
    if (layer == 3) {
      final.swap(sized);
    } else {
      std::vector<double> raw_scalar(nodes * 128);
      equivariant.resize(nodes * 384);
      for (std::size_t n = 0; n < nodes; ++n) {
        std::copy_n(sized.data() + n * 512, 128, raw_scalar.data() + n * 128);
        std::copy_n(sized.data() + n * 512 + 128, 384,
                    equivariant.data() + n * 384);
      }
      condition(f, raw_scalar, scalar);
    }
  }
  if (final.size() != nodes * 128 || edges == 0)
    throw std::runtime_error("invalid final pipeline shape");
}

// Synthetic production-scale workload.
//
// The captured fixture is a correctness artifact: four structures, 77 atoms,
// 1,286 edges. The batch HydraGNN actually trains on in the SC26 configuration
// is 128 structures, 2,726 atoms, and 45,974 directed edges, which is roughly
// 36x larger. Storing that batch's checkpoint tensors would cost multiple
// gigabytes, so this builds a graph at those dimensions and reuses the captured
// model parameters, which are per-species and per-layer and therefore
// independent of how many atoms the batch holds.
//
// The result carries no reference outputs, so it is a timing workload only.
// Correctness is established beforehand on the captured fixture.
namespace production {

// Dimensions of the published SC26 batch. src/HYDRAGNN_SC26.md records the
// profile they come from.
struct Spec {
  std::size_t graphs = 128;
  std::size_t nodes = 2726;
  std::size_t edges = 45974;
};

struct Workload {
  Fixture fixture;
  std::size_t graphs = 0;
  std::size_t nodes = 0;
  std::size_t edges = 0;
  std::size_t requested_edges = 0;
  // Hash of the host-generated topology, in the receiver-grouped order the
  // device flatten step produces. Lets a backend confirm it rebuilt the same
  // graph at production scale, where the quadratic capture-order remap the
  // captured fixture uses would be far too slow.
  std::uint64_t topology_hash = 0;
};

namespace detail {

// All four ports compile this header, and they must agree on the geometry down
// to the last bit, so the sampler is a fixed LCG seeded from the graph index
// rather than anything drawn from <random>, whose engines are portable but
// whose distributions are not.
struct Random {
  std::uint64_t state;

  explicit Random(std::uint64_t seed) : state(seed) { advance(); advance(); }

  void advance() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  }

  double unit() {
    advance();
    return static_cast<double>(state >> 11) / 9007199254740992.0;
  }
};

// One synthetic structure. Fractional coordinates are fixed at construction
// and the cubic cell edge is the only free parameter, so scaling `side` scales
// every interatomic distance uniformly. Neighbor count is therefore monotone
// non-increasing in `side`, which is what lets the edge-count fit below use
// bisection instead of a search over coordinates.
struct Structure {
  std::vector<double> fractional;
  std::size_t atoms = 0;
  double side = 0.0;
};

// Rejection-sampled positions with a minimum separation, so no pair sits at a
// distance the radial basis was never trained on.
inline std::vector<double> sample_fractional(std::size_t atoms,
                                             double min_separation,
                                             std::uint64_t seed) {
  Random random(seed);
  const double min_squared = min_separation * min_separation;
  std::vector<double> result;
  result.reserve(3 * atoms);
  for (std::size_t placed = 0; placed < atoms; ++placed) {
    bool accepted = false;
    for (int attempt = 0; attempt < 100000 && !accepted; ++attempt) {
      const double x = random.unit(), y = random.unit(), z = random.unit();
      accepted = true;
      for (std::size_t i = 0; i < result.size(); i += 3) {
        double dx = x - result[i];
        double dy = y - result[i + 1];
        double dz = z - result[i + 2];
        // Minimum image in fractional space; the cell is cubic.
        dx -= std::round(dx);
        dy -= std::round(dy);
        dz -= std::round(dz);
        if (dx * dx + dy * dy + dz * dz < min_squared) {
          accepted = false;
          break;
        }
      }
      if (accepted) {
        result.push_back(x);
        result.push_back(y);
        result.push_back(z);
      }
    }
    if (!accepted)
      throw std::runtime_error(
          "production workload: could not place atoms at the requested "
          "minimum separation");
  }
  return result;
}

inline mace_nlist::Fixture as_nlist_fixture(const Structure &structure) {
  mace_nlist::Fixture fixture;
  fixture.graph_ptr = {0, static_cast<int>(structure.atoms)};
  fixture.positions.resize(3 * structure.atoms);
  for (std::size_t i = 0; i < fixture.positions.size(); ++i)
    fixture.positions[i] = structure.fractional[i] * structure.side;
  fixture.cells = {structure.side, 0.0, 0.0, 0.0, structure.side, 0.0,
                   0.0, 0.0, structure.side};
  fixture.pbc = {1, 1, 1};
  return fixture;
}

// Runs the shared host neighbor reference over one structure. Returning the
// candidate slots lets the caller flatten them into edges without repeating
// the search.
inline std::size_t neighbors(const Structure &structure,
                            std::vector<mace_nlist::Candidate> *slots,
                            std::vector<int> *counts) {
  const mace_nlist::Fixture fixture = as_nlist_fixture(structure);
  const std::vector<double> inverses = mace_nlist::cell_inverses(fixture);
  std::vector<mace_nlist::Candidate> best(
      structure.atoms * mace_nlist::kMaxNeighbors);
  std::vector<int> found(structure.atoms);
  std::size_t total = 0;
  for (std::size_t dst = 0; dst < structure.atoms; ++dst) {
    int count = 0;
    mace_nlist::reference_atom(fixture, inverses, 0, static_cast<int>(dst),
                               &best[dst * mace_nlist::kMaxNeighbors], count);
    found[dst] = count;
    total += static_cast<std::size_t>(count);
  }
  if (slots) *slots = std::move(best);
  if (counts) *counts = std::move(found);
  return total;
}

inline std::size_t degree(const Structure &structure) {
  return neighbors(structure, nullptr, nullptr);
}

// Smallest cell edge whose total edge count does not exceed `target`. Bisection
// is valid because every structure shares `side` here and degree is monotone
// in it. Sixty halvings take the bracket below double precision.
inline double fit_uniform_side(std::vector<Structure> &structures,
                               std::size_t target, double low, double high) {
  auto total = [&structures](double side) {
    std::size_t sum = 0;
    for (Structure &structure : structures) {
      structure.side = side;
      sum += degree(structure);
    }
    return sum;
  };
  if (total(low) <= target) return low;
  if (total(high) > target)
    throw std::runtime_error(
        "production workload: edge target unreachable within the cell range");
  for (int iteration = 0; iteration < 60; ++iteration) {
    const double middle = 0.5 * (low + high);
    if (total(middle) > target) low = middle; else high = middle;
  }
  total(high);
  return high;
}

// Shrinks one structure to the next larger neighbor count, provided the step
// does not overshoot `limit`. Returns the edges gained, or zero if the
// structure cannot help. Each step usually adds the two directed edges of a
// single pair crossing the cutoff.
inline std::size_t take_step(Structure &structure, std::size_t limit) {
  const std::size_t base = degree(structure);
  double high = structure.side;
  double low = structure.side * 0.75;
  Structure probe = structure;
  probe.side = low;
  if (degree(probe) <= base) return 0;
  for (int iteration = 0; iteration < 60; ++iteration) {
    const double middle = 0.5 * (low + high);
    probe.side = middle;
    if (degree(probe) > base) low = middle; else high = middle;
  }
  probe.side = low;
  const std::size_t gained = degree(probe) - base;
  if (gained > limit) return 0;
  structure.side = low;
  return gained;
}

// Repeats the rows of `base` until `rows` of them exist, so the synthetic batch
// reuses the captured element mix and graph attributes rather than inventing
// values the model never saw.
inline Tensor cycle_rows(const Tensor &base, std::size_t rows) {
  if (base.shape.empty() || base.shape[0] == 0)
    throw std::runtime_error("production workload: cannot cycle an empty tensor");
  const std::size_t base_rows = static_cast<std::size_t>(base.shape[0]);
  const std::size_t width = base.size() / base_rows;
  Tensor result;
  result.shape = base.shape;
  result.shape[0] = rows;
  if (!base.f64.empty()) {
    result.f64.resize(rows * width);
    for (std::size_t row = 0; row < rows; ++row)
      std::copy_n(base.f64.data() + (row % base_rows) * width, width,
                  result.f64.data() + row * width);
  } else {
    result.u64.resize(rows * width);
    for (std::size_t row = 0; row < rows; ++row)
      std::copy_n(base.u64.data() + (row % base_rows) * width, width,
                  result.u64.data() + row * width);
  }
  return result;
}

}  // namespace detail

// Builds the synthetic workload from a captured fixture. Model parameters are
// copied verbatim; geometry, topology, and per-node/per-graph inputs are
// generated at the requested dimensions. Reference tensors are dropped, since
// no captured output corresponds to this input.
inline Workload generate(const Fixture &base, const Spec &spec = Spec()) {
  if (!spec.graphs || spec.nodes < spec.graphs || !spec.edges)
    throw std::runtime_error("production workload: invalid dimensions");

  // Atom counts differ by at most one, matching the captured batch's spread of
  // small molecular and periodic structures.
  std::vector<std::size_t> atoms(spec.graphs, spec.nodes / spec.graphs);
  for (std::size_t g = 0; g < spec.nodes % spec.graphs; ++g) ++atoms[g];

  // Cell edge that would give the requested average degree for a uniform
  // system, used to seed the bisection and to set the minimum separation.
  const double average_degree =
      static_cast<double>(spec.edges) / static_cast<double>(spec.nodes);
  const double sphere = 4.0 / 3.0 * 3.14159265358979323846 *
                        mace_nlist::kCutoff * mace_nlist::kCutoff *
                        mace_nlist::kCutoff;
  const double mean_atoms =
      static_cast<double>(spec.nodes) / static_cast<double>(spec.graphs);
  const double estimate = std::cbrt(sphere * mean_atoms / average_degree);

  std::vector<detail::Structure> structures(spec.graphs);
  for (std::size_t g = 0; g < spec.graphs; ++g) {
    structures[g].atoms = atoms[g];
    structures[g].fractional = detail::sample_fractional(
        atoms[g], 1.2 / estimate, 0x9e3779b97f4a7c15ULL * (g + 1));
    structures[g].side = estimate;
  }

  detail::fit_uniform_side(structures, spec.edges, 0.25 * estimate,
                           4.0 * estimate);
  std::size_t total = 0;
  for (const detail::Structure &structure : structures)
    total += detail::degree(structure);

  // The uniform fit lands just under the target because the total steps as the
  // shared cell edge crosses a pair distance. Close the remainder by shrinking
  // individual structures, which moves the total in steps of about two.
  for (int round = 0; round < 8 && total < spec.edges; ++round)
    for (std::size_t g = 0; g < spec.graphs && total < spec.edges; ++g)
      total += detail::take_step(structures[g], spec.edges - total);

  Fixture fixture;
  fixture.version = base.version < 2 ? 2 : base.version;
  for (const auto &entry : base.tensors) {
    const std::string &name = entry.first;
    if (name.rfind("expected_", 0) == 0) continue;
    if (name == "positions" || name == "cells" || name == "pbc" ||
        name == "graph_ptr" || name == "node_batch" || name == "graph_attr" ||
        name == "raw_x" || name == "edge_index" || name == "edge_shifts" ||
        name == "edge_attr_raw")
      continue;
    fixture.tensors.emplace(name, entry.second);
  }

  Tensor positions, cells, pbc, graph_ptr, node_batch;
  Tensor edge_index, edge_shifts;
  positions.shape = {spec.nodes, 3};
  positions.f64.reserve(3 * spec.nodes);
  cells.shape = {spec.graphs, 3, 3};
  cells.f64.assign(9 * spec.graphs, 0.0);
  pbc.shape = {spec.graphs, 3};
  pbc.u64.assign(3 * spec.graphs, 1);
  graph_ptr.shape = {spec.graphs + 1};
  graph_ptr.u64.reserve(spec.graphs + 1);
  node_batch.shape = {spec.nodes};
  node_batch.u64.reserve(spec.nodes);
  edge_index.f64.clear();
  edge_shifts.f64.reserve(3 * total);
  edge_index.u64.reserve(2 * total);

  std::vector<mace_nlist::Edge> global_edges;
  global_edges.reserve(total);
  std::size_t base_atom = 0;
  graph_ptr.u64.push_back(0);
  for (std::size_t g = 0; g < spec.graphs; ++g) {
    const detail::Structure &structure = structures[g];
    const double side = structure.side;
    cells.f64[9 * g] = side;
    cells.f64[9 * g + 4] = side;
    cells.f64[9 * g + 8] = side;
    for (std::size_t i = 0; i < 3 * structure.atoms; ++i)
      positions.f64.push_back(structure.fractional[i] * side);
    for (std::size_t i = 0; i < structure.atoms; ++i)
      node_batch.u64.push_back(g);

    std::vector<mace_nlist::Candidate> slots;
    std::vector<int> counts;
    detail::neighbors(structure, &slots, &counts);
    const mace_nlist::Fixture local = detail::as_nlist_fixture(structure);
    const std::vector<mace_nlist::Edge> local_edges =
        mace_nlist::flatten(local, slots, counts);
    for (const mace_nlist::Edge &edge : local_edges) {
      edge_index.u64.push_back(
          static_cast<std::uint64_t>(base_atom + edge.src));
      edge_index.u64.push_back(
          static_cast<std::uint64_t>(base_atom + edge.dst));
      edge_shifts.f64.push_back(edge.sx * side);
      edge_shifts.f64.push_back(edge.sy * side);
      edge_shifts.f64.push_back(edge.sz * side);
      mace_nlist::Edge global = edge;
      global.src = static_cast<int>(base_atom) + edge.src;
      global.dst = static_cast<int>(base_atom) + edge.dst;
      global_edges.push_back(global);
    }
    base_atom += structure.atoms;
    graph_ptr.u64.push_back(base_atom);
  }

  const std::size_t edges = edge_index.u64.size() / 2;
  edge_index.shape = {edges, 2};
  edge_shifts.shape = {edges, 3};

  fixture.tensors.emplace("positions", std::move(positions));
  fixture.tensors.emplace("cells", std::move(cells));
  fixture.tensors.emplace("pbc", std::move(pbc));
  fixture.tensors.emplace("graph_ptr", std::move(graph_ptr));
  fixture.tensors.emplace("node_batch", std::move(node_batch));
  fixture.tensors.emplace("edge_index", std::move(edge_index));
  fixture.tensors.emplace("edge_shifts", std::move(edge_shifts));
  fixture.tensors.emplace("raw_x",
                          detail::cycle_rows(base.at("raw_x"), spec.nodes));
  fixture.tensors.emplace(
      "graph_attr", detail::cycle_rows(base.at("graph_attr"), spec.graphs));

  Workload workload;
  workload.graphs = spec.graphs;
  workload.nodes = spec.nodes;
  workload.edges = edges;
  workload.requested_edges = spec.edges;
  workload.topology_hash = mace_nlist::topology_hash(global_edges);
  workload.fixture = std::move(fixture);
  return workload;
}

}  // namespace production

}  // namespace mace_pipeline

#endif
