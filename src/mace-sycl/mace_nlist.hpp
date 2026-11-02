#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mace_nlist {

constexpr double kCutoff = 5.0;
constexpr double kCutoff2 = kCutoff * kCutoff;
constexpr int kMaxNeighbors = 20;

struct Candidate {
  double r2;
  int src, sx, sy, sz;
};

struct Edge {
  double r2, dx, dy, dz;
  int src, dst, sx, sy, sz;
};

struct Fixture {
  std::vector<int> graph_ptr;
  std::vector<double> positions;
  std::vector<double> cells;
  std::vector<int> pbc;

  int graphs() const { return static_cast<int>(graph_ptr.size()) - 1; }
  int atoms() const { return graph_ptr.empty() ? 0 : graph_ptr.back(); }
};

inline bool better(const Candidate &a, const Candidate &b) {
  if (a.r2 != b.r2) return a.r2 < b.r2;
  if (a.src != b.src) return a.src < b.src;
  if (a.sx != b.sx) return a.sx < b.sx;
  if (a.sy != b.sy) return a.sy < b.sy;
  return a.sz < b.sz;
}

inline void insert_candidate(Candidate *best, int &count, const Candidate &c) {
  if (count == kMaxNeighbors && !better(c, best[count - 1])) return;
  int at = std::min(count, kMaxNeighbors - 1);
  if (count < kMaxNeighbors) ++count;
  while (at > 0 && better(c, best[at - 1])) {
    best[at] = best[at - 1];
    --at;
  }
  best[at] = c;
}

inline void inverse3x3(const double *a, double *inv) {
  const double det =
      a[0] * (a[4] * a[8] - a[5] * a[7]) -
      a[1] * (a[3] * a[8] - a[5] * a[6]) +
      a[2] * (a[3] * a[7] - a[4] * a[6]);
  if (!std::isfinite(det) || std::abs(det) < 1.0e-14)
    throw std::runtime_error("cell is singular or numerically degenerate");
  const double q = 1.0 / det;
  inv[0] = (a[4] * a[8] - a[5] * a[7]) * q;
  inv[1] = (a[2] * a[7] - a[1] * a[8]) * q;
  inv[2] = (a[1] * a[5] - a[2] * a[4]) * q;
  inv[3] = (a[5] * a[6] - a[3] * a[8]) * q;
  inv[4] = (a[0] * a[8] - a[2] * a[6]) * q;
  inv[5] = (a[2] * a[3] - a[0] * a[5]) * q;
  inv[6] = (a[3] * a[7] - a[4] * a[6]) * q;
  inv[7] = (a[1] * a[6] - a[0] * a[7]) * q;
  inv[8] = (a[0] * a[4] - a[1] * a[3]) * q;
}

inline void validate_fixture(const Fixture &f) {
  if (f.graph_ptr.size() < 2 || f.graph_ptr.front() != 0)
    throw std::runtime_error(
        "graph_ptr must begin with zero and contain at least one graph");
  for (std::size_t i = 1; i < f.graph_ptr.size(); ++i)
    if (f.graph_ptr[i] < f.graph_ptr[i - 1])
      throw std::runtime_error("graph_ptr must be nondecreasing");
  if (f.atoms() <= 0 ||
      f.positions.size() != static_cast<std::size_t>(3 * f.atoms()))
    throw std::runtime_error("positions size does not match graph_ptr");
  if (f.cells.size() != static_cast<std::size_t>(9 * f.graphs()))
    throw std::runtime_error("cells size does not match graph count");
  if (f.pbc.size() != static_cast<std::size_t>(3 * f.graphs()))
    throw std::runtime_error("PBC size does not match graph count");
  for (double value : f.positions)
    if (!std::isfinite(value))
      throw std::runtime_error("positions must be finite FP64 values");
  for (int value : f.pbc)
    if (value != 0 && value != 1)
      throw std::runtime_error("PBC values must be zero or one");
  double inverse[9];
  for (int graph = 0; graph < f.graphs(); ++graph)
    inverse3x3(&f.cells[9 * graph], inverse);
}

inline std::vector<int> atom_graphs(const Fixture &f) {
  std::vector<int> result(f.atoms());
  for (int graph = 0; graph < f.graphs(); ++graph)
    std::fill(result.begin() + f.graph_ptr[graph],
              result.begin() + f.graph_ptr[graph + 1], graph);
  return result;
}

inline std::vector<double> cell_inverses(const Fixture &f) {
  std::vector<double> result(9 * f.graphs());
  for (int graph = 0; graph < f.graphs(); ++graph)
    inverse3x3(&f.cells[9 * graph], &result[9 * graph]);
  return result;
}

inline void reference_atom(const Fixture &f,
                           const std::vector<double> &inverses, int graph,
                           int dst, Candidate *best, int &count) {
  count = 0;
  const double *cell = &f.cells[9 * graph];
  const double *inverse = &inverses[9 * graph];
  const double bx =
      kCutoff * std::sqrt(inverse[0] * inverse[0] +
                          inverse[3] * inverse[3] +
                          inverse[6] * inverse[6]);
  const double by =
      kCutoff * std::sqrt(inverse[1] * inverse[1] +
                          inverse[4] * inverse[4] +
                          inverse[7] * inverse[7]);
  const double bz =
      kCutoff * std::sqrt(inverse[2] * inverse[2] +
                          inverse[5] * inverse[5] +
                          inverse[8] * inverse[8]);
  for (int src = f.graph_ptr[graph]; src < f.graph_ptr[graph + 1]; ++src) {
    const double x = f.positions[3 * dst] - f.positions[3 * src];
    const double y = f.positions[3 * dst + 1] - f.positions[3 * src + 1];
    const double z = f.positions[3 * dst + 2] - f.positions[3 * src + 2];
    const double fx = x * inverse[0] + y * inverse[3] + z * inverse[6];
    const double fy = x * inverse[1] + y * inverse[4] + z * inverse[7];
    const double fz = x * inverse[2] + y * inverse[5] + z * inverse[8];
    const int lx = f.pbc[3 * graph]
        ? static_cast<int>(std::ceil(std::nextafter(-fx - bx, -INFINITY))) : 0;
    const int ux = f.pbc[3 * graph]
        ? static_cast<int>(std::floor(std::nextafter(-fx + bx, INFINITY))) : 0;
    const int ly = f.pbc[3 * graph + 1]
        ? static_cast<int>(std::ceil(std::nextafter(-fy - by, -INFINITY))) : 0;
    const int uy = f.pbc[3 * graph + 1]
        ? static_cast<int>(std::floor(std::nextafter(-fy + by, INFINITY))) : 0;
    const int lz = f.pbc[3 * graph + 2]
        ? static_cast<int>(std::ceil(std::nextafter(-fz - bz, -INFINITY))) : 0;
    const int uz = f.pbc[3 * graph + 2]
        ? static_cast<int>(std::floor(std::nextafter(-fz + bz, INFINITY))) : 0;
    for (int sx = lx; sx <= ux; ++sx)
      for (int sy = ly; sy <= uy; ++sy)
        for (int sz = lz; sz <= uz; ++sz) {
          if (src == dst && sx == 0 && sy == 0 && sz == 0) continue;
          Candidate candidate;
          const double dx =
              x + sx * cell[0] + sy * cell[3] + sz * cell[6];
          const double dy =
              y + sx * cell[1] + sy * cell[4] + sz * cell[7];
          const double dz =
              z + sx * cell[2] + sy * cell[5] + sz * cell[8];
          candidate.r2 = dx * dx + dy * dy + dz * dz;
          candidate.src = src;
          candidate.sx = sx;
          candidate.sy = sy;
          candidate.sz = sz;
          if (candidate.r2 <= kCutoff2)
            insert_candidate(best, count, candidate);
        }
  }
}

inline std::vector<Edge> flatten(const Fixture &fixture,
                                 const std::vector<Candidate> &slots,
                                 const std::vector<int> &counts) {
  std::vector<Edge> result;
  for (int dst = 0; dst < static_cast<int>(counts.size()); ++dst) {
    if (counts[dst] < 0 || counts[dst] > kMaxNeighbors)
      throw std::runtime_error("invalid neighbor count");
    for (int index = 0; index < counts[dst]; ++index) {
      const Candidate &candidate = slots[dst * kMaxNeighbors + index];
      const int graph = static_cast<int>(
          std::upper_bound(fixture.graph_ptr.begin(), fixture.graph_ptr.end(),
                           dst) - fixture.graph_ptr.begin()) - 1;
      const double *cell = fixture.cells.data() + 9 * graph;
      const double dx = fixture.positions[3 * dst] -
                        fixture.positions[3 * candidate.src] +
                        candidate.sx * cell[0] + candidate.sy * cell[3] +
                        candidate.sz * cell[6];
      const double dy = fixture.positions[3 * dst + 1] -
                        fixture.positions[3 * candidate.src + 1] +
                        candidate.sx * cell[1] + candidate.sy * cell[4] +
                        candidate.sz * cell[7];
      const double dz = fixture.positions[3 * dst + 2] -
                        fixture.positions[3 * candidate.src + 2] +
                        candidate.sx * cell[2] + candidate.sy * cell[5] +
                        candidate.sz * cell[8];
      result.push_back({candidate.r2, dx, dy, dz, candidate.src, dst, candidate.sx,
                        candidate.sy, candidate.sz});
    }
  }
  return result;
}

inline std::uint64_t topology_hash(const std::vector<Edge> &edges) {
  std::uint64_t hash = 1469598103934665603ULL;
  auto add = [&hash](std::uint32_t value) {
    for (int index = 0; index < 4; ++index) {
      hash ^= (value >> (8 * index)) & 255U;
      hash *= 1099511628211ULL;
    }
  };
  for (const Edge &edge : edges) {
    add(static_cast<std::uint32_t>(edge.src));
    add(static_cast<std::uint32_t>(edge.dst));
    add(static_cast<std::uint32_t>(edge.sx));
    add(static_cast<std::uint32_t>(edge.sy));
    add(static_cast<std::uint32_t>(edge.sz));
  }
  return hash;
}

}  // namespace mace_nlist
