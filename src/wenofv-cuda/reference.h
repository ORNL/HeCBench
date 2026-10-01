// CPU golden reference for the WENO FV dynamical-core tendency benchmark.
// Runs the exact same __host__ __device__ physics used by the CUDA kernels, so
// the comparison isolates the GPU port rather than algorithmic differences.

#ifndef WENOFV_REFERENCE_H
#define WENOFV_REFERENCE_H

#include "kernels.h"
#include <vector>

static void reference(const Grid &g, const float *fields, const float *hy_dens,
                      const float *hy_theta_cells, const float *hy_theta_edges,
                      const float *immersed_prop, float *state_tend) {
  const size_t nfx = (size_t)NUM_STATE * g.nz * g.ny * (g.nx + 1);
  const size_t nfy = (size_t)NUM_STATE * g.nz * (g.ny + 1) * g.nx;
  const size_t nfz = (size_t)NUM_STATE * (g.nz + 1) * g.ny * g.nx;
  const size_t nex = (size_t)g.nz * g.ny * (g.nx + 1);
  const size_t ney = (size_t)g.nz * (g.ny + 1) * g.nx;
  const size_t nez = (size_t)(g.nz + 1) * g.ny * g.nx;

  std::vector<float> p_x(nex), ru_x(nex), p_y(ney), rv_y(ney), p_z(nez),
      rw_z(nez);
  std::vector<float> flux_x(nfx), flux_y(nfy), flux_z(nfz);

  // Acoustic upwinding
  for (int k = 0; k < g.nz; k++)
    for (int j = 0; j < g.ny; j++)
      for (int i = 0; i < g.nx + 1; i++)
        acoustic_x(k, j, i, g, fields, hy_dens, immersed_prop,
                   p_x[x3(k, j, i, g.ny, g.nx + 1)],
                   ru_x[x3(k, j, i, g.ny, g.nx + 1)]);
  for (int k = 0; k < g.nz; k++)
    for (int j = 0; j < g.ny + 1; j++)
      for (int i = 0; i < g.nx; i++)
        acoustic_y(k, j, i, g, fields, hy_dens, immersed_prop,
                   p_y[x3(k, j, i, g.ny + 1, g.nx)],
                   rv_y[x3(k, j, i, g.ny + 1, g.nx)]);
  for (int k = 0; k < g.nz + 1; k++)
    for (int j = 0; j < g.ny; j++)
      for (int i = 0; i < g.nx; i++)
        acoustic_z(k, j, i, g, fields, hy_dens, immersed_prop,
                   p_z[x3(k, j, i, g.ny, g.nx)], rw_z[x3(k, j, i, g.ny, g.nx)]);

  // Advective upwinding / total fluxes
  for (int k = 0; k < g.nz; k++)
    for (int j = 0; j < g.ny; j++)
      for (int i = 0; i < g.nx + 1; i++)
        advect_x(k, j, i, g, fields, hy_theta_cells, immersed_prop,
                 ru_x[x3(k, j, i, g.ny, g.nx + 1)],
                 p_x[x3(k, j, i, g.ny, g.nx + 1)], flux_x.data());
  for (int k = 0; k < g.nz; k++)
    for (int j = 0; j < g.ny + 1; j++)
      for (int i = 0; i < g.nx; i++)
        advect_y(k, j, i, g, fields, hy_theta_cells, immersed_prop,
                 rv_y[x3(k, j, i, g.ny + 1, g.nx)],
                 p_y[x3(k, j, i, g.ny + 1, g.nx)], flux_y.data());
  for (int k = 0; k < g.nz + 1; k++)
    for (int j = 0; j < g.ny; j++)
      for (int i = 0; i < g.nx; i++)
        advect_z(k, j, i, g, fields, hy_theta_edges, immersed_prop,
                 rw_z[x3(k, j, i, g.ny, g.nx)], p_z[x3(k, j, i, g.ny, g.nx)],
                 flux_z.data());

  // Flux divergence + gravity -> tendency
  for (int l = 0; l < NUM_STATE; l++)
    for (int k = 0; k < g.nz; k++)
      for (int j = 0; j < g.ny; j++)
        for (int i = 0; i < g.nx; i++)
          state_tend[x4(l, k, j, i, g.nz, g.ny, g.nx)] =
              tendency(l, k, j, i, g, fields, flux_x.data(), flux_y.data(),
                       flux_z.data());
}

#endif // WENOFV_REFERENCE_H
