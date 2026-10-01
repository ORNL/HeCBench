#include "kernels.h"
#include "reference.h"
#include <chrono>
#include <cuda.h>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define TPB 256

#define GPU_CHECK(call)                                                        \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n", __FILE__, __LINE__,    \
              #call, cudaGetErrorString(err_));                                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void //__launch_bounds__(TPB, 3)
k_acoustic_x(size_t n, Grid g, const float *__restrict__ fields,
             const float *__restrict__ hy_dens,
             const float *__restrict__ imm, float *__restrict__ p_x,
             float *__restrict__ ru_x) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n)
    return;
  int i = t % (g.nx + 1);
  size_t r = t / (g.nx + 1);
  int j = r % g.ny;
  int k = r / g.ny;
  acoustic_x(k, j, i, g, fields, hy_dens, imm, p_x[x3(k, j, i, g.ny, g.nx + 1)],
             ru_x[x3(k, j, i, g.ny, g.nx + 1)]);
}

__global__ void //__launch_bounds__(TPB, 3)
k_acoustic_y(size_t n, Grid g, const float *__restrict__ fields,
             const float *__restrict__ hy_dens,
             const float *__restrict__ imm, float *__restrict__ p_y,
             float *__restrict__ rv_y) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n)
    return;
  int i = t % g.nx;
  size_t r = t / g.nx;
  int j = r % (g.ny + 1);
  int k = r / (g.ny + 1);
  acoustic_y(k, j, i, g, fields, hy_dens, imm, p_y[x3(k, j, i, g.ny + 1, g.nx)],
             rv_y[x3(k, j, i, g.ny + 1, g.nx)]);
}

__global__ void //__launch_bounds__(TPB, 3)
k_acoustic_z(size_t n, Grid g, const float *__restrict__ fields,
             const float *__restrict__ hy_dens,
             const float *__restrict__ imm, float *__restrict__ p_z,
             float *__restrict__ rw_z) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n)
    return;
  int i = t % g.nx;
  size_t r = t / g.nx;
  int j = r % g.ny;
  int k = r / g.ny;
  acoustic_z(k, j, i, g, fields, hy_dens, imm, p_z[x3(k, j, i, g.ny, g.nx)],
             rw_z[x3(k, j, i, g.ny, g.nx)]);
}

__global__ void k_advect_x(size_t n, Grid g, const float *__restrict__ fields,
                           const float *__restrict__ hy_theta_cells,
                           const float *__restrict__ imm,
                           const float *__restrict__ ru_x,
                           const float *__restrict__ p_x,
                           float *__restrict__ flux_x) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n)
    return;
  int i = t % (g.nx + 1);
  size_t r = t / (g.nx + 1);
  int j = r % g.ny;
  int k = r / g.ny;
  size_t e = x3(k, j, i, g.ny, g.nx + 1);
  advect_x(k, j, i, g, fields, hy_theta_cells, imm, ru_x[e], p_x[e], flux_x);
}

__global__ void k_advect_y(size_t n, Grid g, const float *__restrict__ fields,
                           const float *__restrict__ hy_theta_cells,
                           const float *__restrict__ imm,
                           const float *__restrict__ rv_y,
                           const float *__restrict__ p_y,
                           float *__restrict__ flux_y) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n)
    return;
  int i = t % g.nx;
  size_t r = t / g.nx;
  int j = r % (g.ny + 1);
  int k = r / (g.ny + 1);
  size_t e = x3(k, j, i, g.ny + 1, g.nx);
  advect_y(k, j, i, g, fields, hy_theta_cells, imm, rv_y[e], p_y[e], flux_y);
}

__global__ void k_advect_z(size_t n, Grid g, const float *__restrict__ fields,
                           const float *__restrict__ hy_theta_edges,
                           const float *__restrict__ imm,
                           const float *__restrict__ rw_z,
                           const float *__restrict__ p_z,
                           float *__restrict__ flux_z) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n)
    return;
  int i = t % g.nx;
  size_t r = t / g.nx;
  int j = r % g.ny;
  int k = r / g.ny;
  size_t e = x3(k, j, i, g.ny, g.nx);
  advect_z(k, j, i, g, fields, hy_theta_edges, imm, rw_z[e], p_z[e], flux_z);
}

__global__ void k_tendency(size_t n, Grid g, const float *__restrict__ fields,
                           const float *__restrict__ flux_x,
                           const float *__restrict__ flux_y,
                           const float *__restrict__ flux_z,
                           float *__restrict__ state_tend) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n)
    return;
  int i = t % g.nx;
  size_t r = t / g.nx;
  int j = r % g.ny;
  r /= g.ny;
  int k = r % g.nz;
  int l = r / g.nz;
  state_tend[x4(l, k, j, i, g.nz, g.ny, g.nx)] =
      tendency(l, k, j, i, g, fields, flux_x, flux_y, flux_z);
}

static inline int nblk(size_t n) { return (int)((n + TPB - 1) / TPB); }

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: %s <nx> <ny> <nz> <repeat>\n", argv[0]);
    printf("  e.g. %s 256 256 64 100\n", argv[0]);
    return 1;
  }
  Grid g;
  g.nx = atoi(argv[1]);
  g.ny = atoi(argv[2]);
  g.nz = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  g.px = g.nx + 2 * HS;
  g.py = g.ny + 2 * HS;
  g.pz = g.nz + 2 * HS;
  const float dx = 20.f;
  g.cs = 350.f;
  g.grav = 9.81f;
  g.r_dx = 1.f / dx;
  g.r_dy = 1.f / dx;
  g.r_dz = 1.f / dx;

  printf("Grid: %d x %d x %d (interior), WENO order %d, halo %d\n", g.nx, g.ny,
         g.nz, ORD, HS);

  const size_t nfields = (size_t)NV * g.pz * g.py * g.px;
  const size_t ntend = (size_t)NUM_STATE * g.nz * g.ny * g.nx;
  const size_t nprop = (size_t)g.pz * g.py * g.px;
  std::vector<float> h_fields(nfields);
  std::vector<float> h_hy_dens(g.pz), h_hy_theta_cells(g.pz),
      h_hy_theta_edges(g.nz + 1);
  std::vector<float> h_imm(nprop, 0.f);
  std::vector<float> h_tend(ntend), h_tend_ref(ntend);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dR(-0.1f, 0.1f), dUVW(-8.f, 8.f),
      dTh(-2.f, 2.f), dP(-200.f, 200.f);
  for (int v = 0; v < NV; v++) {
    for (int k = 0; k < g.pz; k++)
      for (int j = 0; j < g.py; j++)
        for (int i = 0; i < g.px; i++) {
          float val;
          switch (v) {
          case idR:
            val = dR(rng);
            break;
          case idU:
          case idV:
          case idW:
            val = dUVW(rng);
            break;
          case idT:
            val = dTh(rng);
            break;
          default:
            val = dP(rng);
            break; // idP
          }
          h_fields[fidx(v, k, j, i, g)] = val;
        }
  }
  for (int k = 0; k < g.pz; k++) {
    float z = (k - HS) * dx;
    h_hy_dens[k] = 1.2f * expf(-fmaxf(0.f, z) / 8000.f);
    h_hy_theta_cells[k] = 300.f + 0.01f * z;
  }
  for (int k = 0; k <= g.nz; k++)
    h_hy_theta_edges[k] = 300.f + 0.01f * (k * dx);

  // Synthesize a rectangular building obstacle centered in x/y, sitting on the
  // ground. Immersed proportion is 1 inside the box, 0 outside (interior
  // region, offset by halos).
  const int bx0 = g.nx / 2 - g.nx / 8, bx1 = g.nx / 2 + g.nx / 8;
  const int by0 = g.ny / 2 - g.ny / 8, by1 = g.ny / 2 + g.ny / 8;
  const int bz1 = g.nz / 4;
  size_t nimm = 0;
  for (int k = 0; k < g.nz; k++)
    for (int j = 0; j < g.ny; j++)
      for (int i = 0; i < g.nx; i++) {
        if (i >= bx0 && i < bx1 && j >= by0 && j < by1 && k < bz1) {
          h_imm[pidx(HS + k, HS + j, HS + i, g)] = 1.f;
          nimm++;
        }
      }
  printf("Immersed building: %zu / %d cells (%.1f%%)\n", nimm,
         g.nx * g.ny * g.nz, 100.0 * nimm / ((double)g.nx * g.ny * g.nz));

  reference(g, h_fields.data(), h_hy_dens.data(), h_hy_theta_cells.data(),
            h_hy_theta_edges.data(), h_imm.data(), h_tend_ref.data());

  const size_t nfx = (size_t)NUM_STATE * g.nz * g.ny * (g.nx + 1);
  const size_t nfy = (size_t)NUM_STATE * g.nz * (g.ny + 1) * g.nx;
  const size_t nfz = (size_t)NUM_STATE * (g.nz + 1) * g.ny * g.nx;
  const size_t nex = (size_t)g.nz * g.ny * (g.nx + 1);
  const size_t ney = (size_t)g.nz * (g.ny + 1) * g.nx;
  const size_t nez = (size_t)(g.nz + 1) * g.ny * g.nx;

  float *d_fields, *d_hy_dens, *d_hy_tc, *d_hy_te, *d_imm;
  float *d_px, *d_rux, *d_py, *d_rvy, *d_pz, *d_rwz;
  float *d_fx, *d_fy, *d_fz, *d_tend;
  GPU_CHECK(cudaMalloc(&d_fields, nfields * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_hy_dens, g.pz * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_hy_tc, g.pz * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_hy_te, (g.nz + 1) * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_imm, nprop * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_px, nex * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_rux, nex * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_py, ney * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_rvy, ney * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_pz, nez * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_rwz, nez * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_fx, nfx * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_fy, nfy * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_fz, nfz * sizeof(float)));
  GPU_CHECK(cudaMalloc(&d_tend, ntend * sizeof(float)));

  GPU_CHECK(cudaMemcpy(d_fields, h_fields.data(), nfields * sizeof(float),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_hy_dens, h_hy_dens.data(), g.pz * sizeof(float),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_hy_tc, h_hy_theta_cells.data(), g.pz * sizeof(float),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_hy_te, h_hy_theta_edges.data(),
                       (g.nz + 1) * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(d_imm, h_imm.data(), nprop * sizeof(float),
                       cudaMemcpyHostToDevice));

  // Warmup
  const int warmup = 100;
  for (int r = 0; r < warmup; r++) {
    k_acoustic_x<<<nblk(nex), TPB>>>(nex, g, d_fields, d_hy_dens, d_imm, d_px,
                                     d_rux);
    k_acoustic_y<<<nblk(ney), TPB>>>(ney, g, d_fields, d_hy_dens, d_imm, d_py,
                                     d_rvy);
    k_acoustic_z<<<nblk(nez), TPB>>>(nez, g, d_fields, d_hy_dens, d_imm, d_pz,
                                     d_rwz);
    k_advect_x<<<nblk(nex), TPB>>>(nex, g, d_fields, d_hy_tc, d_imm, d_rux,
                                   d_px, d_fx);
    k_advect_y<<<nblk(ney), TPB>>>(ney, g, d_fields, d_hy_tc, d_imm, d_rvy,
                                   d_py, d_fy);
    k_advect_z<<<nblk(nez), TPB>>>(nez, g, d_fields, d_hy_te, d_imm, d_rwz,
                                   d_pz, d_fz);
    k_tendency<<<nblk(ntend), TPB>>>(ntend, g, d_fields, d_fx, d_fy, d_fz,
                                     d_tend);
  }

  GPU_CHECK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    k_acoustic_x<<<nblk(nex), TPB>>>(nex, g, d_fields, d_hy_dens, d_imm, d_px,
                                     d_rux);
    k_acoustic_y<<<nblk(ney), TPB>>>(ney, g, d_fields, d_hy_dens, d_imm, d_py,
                                     d_rvy);
    k_acoustic_z<<<nblk(nez), TPB>>>(nez, g, d_fields, d_hy_dens, d_imm, d_pz,
                                     d_rwz);
    k_advect_x<<<nblk(nex), TPB>>>(nex, g, d_fields, d_hy_tc, d_imm, d_rux,
                                   d_px, d_fx);
    k_advect_y<<<nblk(ney), TPB>>>(ney, g, d_fields, d_hy_tc, d_imm, d_rvy,
                                   d_py, d_fy);
    k_advect_z<<<nblk(nez), TPB>>>(nez, g, d_fields, d_hy_te, d_imm, d_rwz,
                                   d_pz, d_fz);
    k_tendency<<<nblk(ntend), TPB>>>(ntend, g, d_fields, d_fx, d_fy, d_fz,
                                     d_tend);
  }
  GPU_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() *
              1e-3;
  printf("Average execution time of compute_tendencies (7 kernels): %f (us)\n",
         us / repeat);

  GPU_CHECK(cudaMemcpy(h_tend.data(), d_tend, ntend * sizeof(float),
                       cudaMemcpyDeviceToHost));

  // ------------------------------------------------ verify
  double max_rel = 0.0;
  size_t nbad = 0;
  for (size_t n = 0; n < ntend; n++) {
    float a = h_tend[n], b = h_tend_ref[n];
    double denom = fmax(1e-4, fmax(fabs((double)a), fabs((double)b)));
    double rel = fabs((double)a - (double)b) / denom;
    if (rel > max_rel)
      max_rel = rel;
    if (rel > 1e-2)
      nbad++;
  }
  printf("Max relative error: %e (%zu / %zu elements exceed 1e-2)\n", max_rel,
         nbad, ntend);
  printf("%s\n", nbad == 0 ? "PASS" : "FAIL");

  GPU_CHECK(cudaFree(d_fields));
  GPU_CHECK(cudaFree(d_hy_dens));
  GPU_CHECK(cudaFree(d_hy_tc));
  GPU_CHECK(cudaFree(d_hy_te));
  GPU_CHECK(cudaFree(d_imm));
  GPU_CHECK(cudaFree(d_px));
  GPU_CHECK(cudaFree(d_rux));
  GPU_CHECK(cudaFree(d_py));
  GPU_CHECK(cudaFree(d_rvy));
  GPU_CHECK(cudaFree(d_pz));
  GPU_CHECK(cudaFree(d_rwz));
  GPU_CHECK(cudaFree(d_fx));
  GPU_CHECK(cudaFree(d_fy));
  GPU_CHECK(cudaFree(d_fz));
  GPU_CHECK(cudaFree(d_tend));
  return 0;
}
