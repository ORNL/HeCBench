// Standalone extraction of Condor::impl::Calc::T_host.
//
// Source: ORNL/Condor commit ca3c4baecbfb04876270b9bb6b180b742eb4a948
//   include/impl/Calc/Temperature.cpp
//   include/impl/Utility/Grid.hpp
//
// The interface uses this benchmark's plain C++ containers, but the index
// conversion, coordinate reconstruction, operation order, and expression are
// kept the same as the upstream host reference.

#pragma once

#include <cmath>
#include <cstddef>

namespace condor_reference {

// Recover i, the slowest grid index, from Condor's linear point index.
inline int p_to_i(const int p, const int y_num, const int z_num) {
  return p / (y_num * z_num);
}

// Recover j, the index across the track, from Condor's linear point index.
inline int p_to_j(const int p, const int y_num, const int z_num) {
  return (p / z_num) % y_num;
}

// Recover k, the depth index, from Condor's linear point index.
inline int p_to_k(const int p, const int z_num) {
  return p % z_num;
}

// Host temperature at one grid point: rebuild its coordinates, then add every
// beam snapshot. The index conversion and operation order match Condor's T_host.
template <typename Nodes, typename Grid, typename FloatType>
FloatType temperature(const Nodes& nodes, const Grid& grid, const FloatType t_init, const int p) {
  const int i = p_to_i(p, grid.ny, grid.nz);
  const int j = p_to_j(p, grid.ny, grid.nz);
  const int k = p_to_k(p, grid.nz);
  const FloatType x0 = (grid.nx == 1) ? grid.xmax : grid.xmin;
  const FloatType y0 = (grid.ny == 1) ? grid.ymax : grid.ymin;
  const FloatType z0 = (grid.nz == 1) ? grid.zmax : grid.zmin;
  const FloatType xp = x0 + static_cast<FloatType>(i) * grid.xres;
  const FloatType yp = y0 + static_cast<FloatType>(j) * grid.yres;
  const FloatType zp = z0 + static_cast<FloatType>(k) * grid.zres;

  FloatType dT = static_cast<FloatType>(0.0);
  const std::size_t num_nodes = nodes.xb.size();
  for (std::size_t node = 0; node < num_nodes; ++node) {
    const FloatType dx = xp - nodes.xb[node];
    const FloatType dy = yp - nodes.yb[node];
    const FloatType dz = zp - nodes.zb[node];

    const FloatType phix = nodes.phix[node];
    const FloatType phiy = nodes.phiy[node];
    const FloatType phiz = nodes.phiz[node];
    const FloatType expmod = nodes.expmod[node];
    const FloatType dtau = nodes.dtau[node];

    const FloatType phi =
        std::exp(-static_cast<FloatType>(3.0) *
                     ((dx * dx * phix) + (dy * dy * phiy) + (dz * dz * phiz)) +
                 expmod);
    const FloatType dT_seg = dtau * phi;
    dT += dT_seg;
  }

  return t_init + dT;
}

}  // namespace condor_reference
