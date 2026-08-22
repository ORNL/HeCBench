#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <new>
#include <vector>
#include <cuda.h>
#include "reference.h"

#define CHECK(call)                                                          \
  do {                                                                       \
    const cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d '%s': %s\n", __FILE__, __LINE__,     \
              #call, cudaGetErrorString(err));                               \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#ifndef P2G_BLOCK
#define P2G_BLOCK 128
#endif
#ifndef GRID_BLOCK
#define GRID_BLOCK 256
#endif
#ifndef PREPROCESS_BLOCK
#define PREPROCESS_BLOCK 128
#endif

// ---------------------------------------------------------------------------
// stage 1: MLS-MPM
//
// The per particle 3x3 tensors are stored component major, so that the nine
// loads of a warp are nine fully coalesced 128 byte transactions instead of
// nine strided ones. The particles are pre-sorted by cell (see
// generate_scene), which keeps the scattered atomics of a warp inside a
// handful of cache lines.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void quad_weights(float fx, float& w0, float& w1,
                                             float& w2)
{
  const float a = 1.5f - fx;
  const float b = fx - 1.0f;
  const float c = fx - 0.5f;
  w0 = 0.5f * a * a;
  w1 = 0.75f - b * b;
  w2 = 0.5f * c * c;
}

// One thread block scatters one chunk of particles, all of which belong to the
// same MPM_BLOCK^3 cell block. Their entire stencil footprint therefore fits a
// MPM_TILE^3 tile held in shared memory, so the scatter costs shared atomics
// plus one flush of the tile, rather than 108 global atomics per particle.
// A particle that has drifted out of its block since the binning still lands
// correctly through the global fallback.
__global__ void __launch_bounds__(P2G_BLOCK)
mpm_p2g_kernel(int n, MpmParams p,
               const int* __restrict__ chunk_start,
               const int* __restrict__ chunk_block,
               const float4* __restrict__ mean,
               const float4* __restrict__ velocity,
               const float* __restrict__ affine_in,
               const float* __restrict__ defgrad,
               float* __restrict__ grid)
{
  __shared__ float tile[MPM_TILE_CELLS * 4];

  for (int k = threadIdx.x; k < MPM_TILE_CELLS * 4; k += P2G_BLOCK) tile[k] = 0.0f;

  const int chunk = blockIdx.x;
  const int begin = chunk_start[chunk];
  const int end = chunk_start[chunk + 1];

  const int gb = chunk_block[chunk];
  const int obz = (gb % MPM_BLOCKS_PER_DIM) * MPM_BLOCK - 1;
  const int oby = ((gb / MPM_BLOCKS_PER_DIM) % MPM_BLOCKS_PER_DIM) * MPM_BLOCK - 1;
  const int obx = (gb / (MPM_BLOCKS_PER_DIM * MPM_BLOCKS_PER_DIM)) * MPM_BLOCK - 1;

  __syncthreads();

  for (int i = begin + threadIdx.x; i < end; i += P2G_BLOCK) {
  const float4 x = mean[i];
  const float4 v = velocity[i];

  float C[9], F0[9];
  #pragma unroll
  for (int k = 0; k < 9; k++) C[k] = affine_in[(size_t)k * n + i];
  #pragma unroll
  for (int k = 0; k < 9; k++) F0[k] = defgrad[(size_t)k * n + i];

  // F <- (I + dt C) F
  float F[9];
  #pragma unroll
  for (int a = 0; a < 3; a++)
    #pragma unroll
    for (int b = 0; b < 3; b++) {
      float acc = F0[a * 3 + b];
      #pragma unroll
      for (int k = 0; k < 3; k++)
        acc = fmaf(p.dt * C[a * 3 + k], F0[k * 3 + b], acc);
      F[a * 3 + b] = acc;
    }

  // neo-Hookean stress, folded into the MLS-MPM affine momentum
  const float det =
      F[0] * (F[4] * F[8] - F[5] * F[7]) -
      F[1] * (F[3] * F[8] - F[5] * F[6]) +
      F[2] * (F[3] * F[7] - F[4] * F[6]);
  const float safe_J = (fabsf(det) < 1e-6f) ? ((det < 0.0f) ? -1e-6f : 1e-6f) : det;
  const float inv_J = 1.0f / safe_J;

  const float cof[9] = {
     (F[4] * F[8] - F[5] * F[7]), -(F[3] * F[8] - F[5] * F[6]),  (F[3] * F[7] - F[4] * F[6]),
    -(F[1] * F[8] - F[2] * F[7]),  (F[0] * F[8] - F[2] * F[6]), -(F[0] * F[7] - F[1] * F[6]),
     (F[1] * F[5] - F[2] * F[4]), -(F[0] * F[5] - F[2] * F[3]),  (F[0] * F[4] - F[1] * F[3]) };

  const float coeff = (p.lambda * __logf(fabsf(safe_J)) - p.mu) * inv_J;
  float P[9];
  #pragma unroll
  for (int k = 0; k < 9; k++) P[k] = fmaf(coeff, cof[k], p.mu * F[k]);

  const float s = -p.dt * p.particle_volume * 4.0f * p.inv_dx * p.inv_dx;
  float affine[9];
  #pragma unroll
  for (int a = 0; a < 3; a++)
    #pragma unroll
    for (int b = 0; b < 3; b++) {
      float acc = 0.0f;
      #pragma unroll
      for (int k = 0; k < 3; k++) acc = fmaf(P[a * 3 + k], F[b * 3 + k], acc);
      affine[a * 3 + b] = fmaf(s, acc, p.particle_mass * C[a * 3 + b]);
    }

  // F is only used for the stress above; it is deliberately not persisted
  // here. The single per-step deformation-gradient update is applied once, in
  // mpm_g2p, from the original F0 (this matches the host reference; writing F
  // back here would advance the gradient twice per step).

  const float gx = x.x * p.inv_dx, gy = x.y * p.inv_dx, gz = x.z * p.inv_dx;
  const int bx = (int)floorf(gx - 0.5f);
  const int by = (int)floorf(gy - 0.5f);
  const int bz = (int)floorf(gz - 0.5f);
  const float fx = gx - bx, fy = gy - by, fz = gz - bz;

  float wx[3], wy[3], wz[3];
  quad_weights(fx, wx[0], wx[1], wx[2]);
  quad_weights(fy, wy[0], wy[1], wy[2]);
  quad_weights(fz, wz[0], wz[1], wz[2]);

  const float mv[3] = { p.particle_mass * v.x, p.particle_mass * v.y,
                        p.particle_mass * v.z };

  #pragma unroll
  for (int a = 0; a < 3; a++) {
    const int ix = bx + a;
    if (ix < 0 || ix >= MPM_GRID) continue;
    const float dpx = ((float)a - fx) * p.dx;
    #pragma unroll
    for (int b = 0; b < 3; b++) {
      const int iy = by + b;
      if (iy < 0 || iy >= MPM_GRID) continue;
      const float dpy = ((float)b - fy) * p.dx;
      const float wxy = wx[a] * wy[b];
      #pragma unroll
      for (int c = 0; c < 3; c++) {
        const int iz = bz + c;
        if (iz < 0 || iz >= MPM_GRID) continue;
        const float dpz = ((float)c - fz) * p.dx;
        const float w = wxy * wz[c];

        const int lx = ix - obx, ly = iy - oby, lz = iz - obz;
        const bool local = (unsigned)lx < MPM_TILE && (unsigned)ly < MPM_TILE &&
                           (unsigned)lz < MPM_TILE;
        float* cell = local
            ? tile + 4 * ((lx * MPM_TILE + ly) * MPM_TILE + lz)
            : grid + 4 * ((size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz);

        #pragma unroll
        for (int k = 0; k < 3; k++) {
          float impulse = affine[k * 3 + 0] * dpx;
          impulse = fmaf(affine[k * 3 + 1], dpy, impulse);
          impulse = fmaf(affine[k * 3 + 2], dpz, impulse);
          atomicAdd(cell + k, w * (mv[k] + impulse));
        }
        atomicAdd(cell + 3, w * p.particle_mass);
      }
    }
  }
  }

  __syncthreads();

  // flush the tile into the global grid
  for (int c = threadIdx.x; c < MPM_TILE_CELLS; c += P2G_BLOCK) {
    const int lz = c % MPM_TILE;
    const int ly = (c / MPM_TILE) % MPM_TILE;
    const int lx = c / (MPM_TILE * MPM_TILE);
    const int ix = obx + lx, iy = oby + ly, iz = obz + lz;
    if ((unsigned)ix >= MPM_GRID || (unsigned)iy >= MPM_GRID ||
        (unsigned)iz >= MPM_GRID)
      continue;

    const float4 acc = *(const float4*)(tile + 4 * c);
    if (acc.x == 0.0f && acc.y == 0.0f && acc.z == 0.0f && acc.w == 0.0f) continue;

    float* dst = grid + 4 * ((size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz);
    atomicAdd(dst + 0, acc.x);
    atomicAdd(dst + 1, acc.y);
    atomicAdd(dst + 2, acc.z);
    atomicAdd(dst + 3, acc.w);
  }
}

__global__ void __launch_bounds__(GRID_BLOCK)
mpm_grid_kernel(MpmParams p, float4* __restrict__ grid)
{
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= MPM_CELLS) return;

  float4 g = grid[cell];
  if (g.w <= 0.0f) {
    if (g.x != 0.0f || g.y != 0.0f || g.z != 0.0f)
      grid[cell] = make_float4(0.0f, 0.0f, 0.0f, g.w);
    return;
  }

  const float inv_mass = 1.0f / g.w;
  float vx = g.x * inv_mass;
  float vy = fmaf(p.dt, p.gravity, g.y * inv_mass);
  float vz = g.z * inv_mass;

  const int iz = cell % MPM_GRID;
  const int iy = (cell / MPM_GRID) % MPM_GRID;
  const int ix = cell / (MPM_GRID * MPM_GRID);

  if (ix < p.boundary && vx < 0.0f) vx = 0.0f;
  if (ix >= MPM_GRID - p.boundary && vx > 0.0f) vx = 0.0f;
  if (iy < p.boundary && vy < 0.0f) vy = 0.0f;
  if (iy >= MPM_GRID - p.boundary && vy > 0.0f) vy = 0.0f;
  if (iz < p.boundary && vz < 0.0f) vz = 0.0f;
  if (iz >= MPM_GRID - p.boundary && vz > 0.0f) vz = 0.0f;

  grid[cell] = make_float4(vx, vy, vz, g.w);
}

__global__ void __launch_bounds__(P2G_BLOCK)
mpm_g2p_kernel(int n, MpmParams p,
               float4* __restrict__ mean,
               float4* __restrict__ velocity,
               float* __restrict__ affine_out,
               float* __restrict__ defgrad,
               const float4* __restrict__ grid)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float4 x = mean[i];
  const float gx = x.x * p.inv_dx, gy = x.y * p.inv_dx, gz = x.z * p.inv_dx;
  const int bx = (int)floorf(gx - 0.5f);
  const int by = (int)floorf(gy - 0.5f);
  const int bz = (int)floorf(gz - 0.5f);
  const float fx = gx - bx, fy = gy - by, fz = gz - bz;

  float wx[3], wy[3], wz[3];
  quad_weights(fx, wx[0], wx[1], wx[2]);
  quad_weights(fy, wy[0], wy[1], wy[2]);
  quad_weights(fz, wz[0], wz[1], wz[2]);

  float nv[3] = { 0.0f, 0.0f, 0.0f };
  float nC[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

  #pragma unroll
  for (int a = 0; a < 3; a++) {
    const int ix = bx + a;
    if (ix < 0 || ix >= MPM_GRID) continue;
    const float dpx = ((float)a - fx) * p.dx;
    #pragma unroll
    for (int b = 0; b < 3; b++) {
      const int iy = by + b;
      if (iy < 0 || iy >= MPM_GRID) continue;
      const float dpy = ((float)b - fy) * p.dx;
      const float wxy = wx[a] * wy[b];
      #pragma unroll
      for (int c = 0; c < 3; c++) {
        const int iz = bz + c;
        if (iz < 0 || iz >= MPM_GRID) continue;
        const float w = wxy * wz[c];
        const float4 g = grid[(size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz];
        const float gv[3] = { g.x, g.y, g.z };
        const float dpos[3] = { dpx, dpy, ((float)c - fz) * p.dx };

        #pragma unroll
        for (int k = 0; k < 3; k++) {
          nv[k] = fmaf(w, gv[k], nv[k]);
          const float wg = 4.0f * p.inv_dx * p.inv_dx * w * gv[k];
          #pragma unroll
          for (int l = 0; l < 3; l++) nC[k * 3 + l] = fmaf(wg, dpos[l], nC[k * 3 + l]);
        }
      }
    }
  }

  float F0[9];
  #pragma unroll
  for (int k = 0; k < 9; k++) F0[k] = defgrad[(size_t)k * n + i];

  #pragma unroll
  for (int a = 0; a < 3; a++)
    #pragma unroll
    for (int b = 0; b < 3; b++) {
      float acc = F0[a * 3 + b];
      #pragma unroll
      for (int k = 0; k < 3; k++)
        acc = fmaf(p.dt * nC[a * 3 + k], F0[k * 3 + b], acc);
      defgrad[(size_t)(a * 3 + b) * n + i] = acc;
    }

  #pragma unroll
  for (int k = 0; k < 9; k++) affine_out[(size_t)k * n + i] = nC[k];

  velocity[i] = make_float4(nv[0], nv[1], nv[2], 0.0f);
  mean[i] = make_float4(fmaf(p.dt, nv[0], x.x), fmaf(p.dt, nv[1], x.y),
                        fmaf(p.dt, nv[2], x.z), x.w);
}

// ---------------------------------------------------------------------------
// stage 2: condition the 4D Gaussian on t, project, shade
//
// The frustum and opacity tests come before the 48 spherical harmonic loads,
// so a culled Gaussian costs no bandwidth beyond its pose.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(PREPROCESS_BLOCK)
preprocess_kernel(int n, float time, Camera cam,
                  const float4* __restrict__ mean4,
                  const float4* __restrict__ scale4,
                  const float4* __restrict__ quat_l,
                  const float4* __restrict__ quat_r,
                  const float* __restrict__ opacity_in,
                  const float* __restrict__ sh,
                  float2* __restrict__ mean2d,
                  float4* __restrict__ conic_opacity,
                  float4* __restrict__ color_depth,
                  int* __restrict__ radii)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  radii[i] = 0;

  const float4 mu = mean4[i];
  const float4 sc = scale4[i];
  const float4 ql = quat_l[i];
  const float4 qr = quat_r[i];

  // M = L(ql) R(qr), the 4D rotation as a pair of isoclinic rotations
  const float lw = ql.x, lx = ql.y, ly = ql.z, lz = ql.w;
  const float rw = qr.x, rx = qr.y, ry = qr.z, rz = qr.w;

  const float L[16] = {
    lw, -lx, -ly, -lz,
    lx,  lw, -lz,  ly,
    ly,  lz,  lw, -lx,
    lz, -ly,  lx,  lw };
  const float R[16] = {
    rw, -rx, -ry, -rz,
    rx,  rw,  rz, -ry,
    ry, -rz,  rw,  rx,
    rz,  ry, -rx,  rw };

  float M[16];
  #pragma unroll
  for (int a = 0; a < 4; a++)
    #pragma unroll
    for (int b = 0; b < 4; b++) {
      float acc = 0.0f;
      #pragma unroll
      for (int k = 0; k < 4; k++) acc = fmaf(L[a * 4 + k], R[k * 4 + b], acc);
      M[a * 4 + b] = acc;
    }

  // the quaternion basis is ordered (t, x, y, z)
  const float s2[4] = { sc.w * sc.w, sc.x * sc.x, sc.y * sc.y, sc.z * sc.z };

  // only the 3x3 spatial block, the spatio-temporal column and Sigma_tt are
  // needed, so the remaining entries of the 4x4 product are never formed
  float st[3], sxyz[6];
  #pragma unroll
  for (int a = 0; a < 3; a++) {
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++) acc = fmaf(M[(a + 1) * 4 + k] * s2[k], M[k], acc);
    st[a] = acc;
  }
  float sigma_tt = 0.0f;
  #pragma unroll
  for (int k = 0; k < 4; k++) sigma_tt = fmaf(M[k] * s2[k], M[k], sigma_tt);

  int idx = 0;
  #pragma unroll
  for (int a = 0; a < 3; a++)
    #pragma unroll
    for (int b = a; b < 3; b++) {
      float acc = 0.0f;
      #pragma unroll
      for (int k = 0; k < 4; k++)
        acc = fmaf(M[(a + 1) * 4 + k] * s2[k], M[(b + 1) * 4 + k], acc);
      sxyz[idx++] = acc;
    }

  const float inv_tt = 1.0f / sigma_tt;
  const float dt = time - mu.w;

  const float alpha = opacity_in[i] * __expf(-0.5f * dt * dt * inv_tt);
  if (alpha < MIN_OPACITY) return;

  const float mean3[3] = { fmaf(dt * inv_tt, st[0], mu.x),
                           fmaf(dt * inv_tt, st[1], mu.y),
                           fmaf(dt * inv_tt, st[2], mu.z) };
  const float cov3[6] = { sxyz[0] - st[0] * st[0] * inv_tt,
                          sxyz[1] - st[0] * st[1] * inv_tt,
                          sxyz[2] - st[0] * st[2] * inv_tt,
                          sxyz[3] - st[1] * st[1] * inv_tt,
                          sxyz[4] - st[1] * st[2] * inv_tt,
                          sxyz[5] - st[2] * st[2] * inv_tt };

  // world -> camera
  const float px = mean3[0] - cam.cam_pos[0];
  const float py = mean3[1] - cam.cam_pos[1];
  const float pz = mean3[2] - cam.cam_pos[2];
  float t[3];
  #pragma unroll
  for (int a = 0; a < 3; a++)
    t[a] = fmaf(cam.view[a * 3 + 0], px,
                fmaf(cam.view[a * 3 + 1], py, cam.view[a * 3 + 2] * pz));

  if (t[2] < cam.near_plane) return;

  const float inv_z = 1.0f / t[2];
  const float inv_z2 = inv_z * inv_z;

  const float lim_x = 1.3f * (0.5f * cam.width) / cam.focal_x;
  const float lim_y = 1.3f * (0.5f * cam.height) / cam.focal_y;
  const float tx = t[2] * fminf(lim_x, fmaxf(-lim_x, t[0] * inv_z));
  const float ty = t[2] * fminf(lim_y, fmaxf(-lim_y, t[1] * inv_z));

  const float J[6] = { cam.focal_x * inv_z, 0.0f, -cam.focal_x * tx * inv_z2,
                       0.0f, cam.focal_y * inv_z, -cam.focal_y * ty * inv_z2 };

  float T[6];
  #pragma unroll
  for (int a = 0; a < 2; a++)
    #pragma unroll
    for (int b = 0; b < 3; b++)
      T[a * 3 + b] = fmaf(J[a * 3 + 0], cam.view[0 * 3 + b],
                          fmaf(J[a * 3 + 1], cam.view[1 * 3 + b],
                               J[a * 3 + 2] * cam.view[2 * 3 + b]));

  const float S[9] = { cov3[0], cov3[1], cov3[2],
                       cov3[1], cov3[3], cov3[4],
                       cov3[2], cov3[4], cov3[5] };
  float TS[6];
  #pragma unroll
  for (int a = 0; a < 2; a++)
    #pragma unroll
    for (int b = 0; b < 3; b++)
      TS[a * 3 + b] = fmaf(T[a * 3 + 0], S[0 * 3 + b],
                           fmaf(T[a * 3 + 1], S[1 * 3 + b],
                                T[a * 3 + 2] * S[2 * 3 + b]));

  float ca = 0.3f, cb = 0.0f, cc = 0.3f;
  #pragma unroll
  for (int k = 0; k < 3; k++) {
    ca = fmaf(TS[0 * 3 + k], T[0 * 3 + k], ca);
    cb = fmaf(TS[0 * 3 + k], T[1 * 3 + k], cb);
    cc = fmaf(TS[1 * 3 + k], T[1 * 3 + k], cc);
  }

  const float det = fmaf(ca, cc, -cb * cb);
  if (det <= 0.0f) return;
  const float inv_det = 1.0f / det;

  const float mid = 0.5f * (ca + cc);
  const float disc = sqrtf(fmaxf(0.1f, fmaf(mid, mid, -det)));
  const float radius = ceilf(3.0f * sqrtf(fmaxf(mid + disc, mid - disc)));

  const float mx = fmaf(cam.focal_x * inv_z, t[0], cam.center_x);
  const float my = fmaf(cam.focal_y * inv_z, t[1], cam.center_y);

  if (mx + radius < 0.0f || mx - radius > (float)cam.width ||
      my + radius < 0.0f || my - radius > (float)cam.height)
    return;

  // spherical harmonics, evaluated only for the Gaussians that survive
  const float inv_len = rsqrtf(fmaf(px, px, fmaf(py, py, pz * pz)));
  const float dx = px * inv_len, dy = py * inv_len, dz = pz * inv_len;

  float basis[SH_COEFFS];
  basis[0] = SH_C0;
  basis[1] = -SH_C1 * dy;
  basis[2] = SH_C1 * dz;
  basis[3] = -SH_C1 * dx;

  const float xx = dx * dx, yy = dy * dy, zz = dz * dz;
  const float xy = dx * dy, yz = dy * dz, xz = dx * dz;
  basis[4] = SH_C2_0 * xy;
  basis[5] = SH_C2_1 * yz;
  basis[6] = SH_C2_2 * (2.0f * zz - xx - yy);
  basis[7] = SH_C2_3 * xz;
  basis[8] = SH_C2_4 * (xx - yy);
  basis[9]  = SH_C3_0 * dy * (3.0f * xx - yy);
  basis[10] = SH_C3_1 * xy * dz;
  basis[11] = SH_C3_2 * dy * (4.0f * zz - xx - yy);
  basis[12] = SH_C3_3 * dz * (2.0f * zz - 3.0f * xx - 3.0f * yy);
  basis[13] = SH_C3_4 * dx * (4.0f * zz - xx - yy);
  basis[14] = SH_C3_5 * dz * (xx - yy);
  basis[15] = SH_C3_6 * dx * (xx - 3.0f * yy);

  float rgb[3] = { 0.0f, 0.0f, 0.0f };
  #pragma unroll
  for (int c = 0; c < SH_COEFFS; c++) {
    const float w = basis[c];
    #pragma unroll
    for (int ch = 0; ch < 3; ch++)
      rgb[ch] = fmaf(w, sh[((size_t)c * 3 + ch) * n + i], rgb[ch]);
  }

  radii[i] = (int)radius;
  mean2d[i] = make_float2(mx, my);
  conic_opacity[i] = make_float4(cc * inv_det, -cb * inv_det, ca * inv_det, alpha);
  color_depth[i] = make_float4(fmaxf(rgb[0] + 0.5f, 0.0f),
                               fmaxf(rgb[1] + 0.5f, 0.0f),
                               fmaxf(rgb[2] + 0.5f, 0.0f), t[2]);
}

// ---------------------------------------------------------------------------
// stage 3: tile rasterizer
//
// One block owns one 16x16 tile. The block cooperatively stages a batch of
// BLOCK_SIZE Gaussians in shared memory, so each Gaussian is read from global
// memory once per tile rather than once per pixel, and the whole block leaves
// as soon as all of its pixels are saturated.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(BLOCK_SIZE)
render_kernel(Camera cam,
              const float2* __restrict__ mean2d,
              const float4* __restrict__ conic_opacity,
              const float4* __restrict__ color_depth,
              const int* __restrict__ tile_offsets,
              const int* __restrict__ tile_list,
              float4* __restrict__ image)
{
  __shared__ float2 s_xy[BLOCK_SIZE];
  __shared__ float4 s_co[BLOCK_SIZE];
  __shared__ float4 s_color[BLOCK_SIZE];

  const int tile = blockIdx.y * cam.tiles_x + blockIdx.x;
  const int x = blockIdx.x * BLOCK_X + threadIdx.x;
  const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
  const int lane = threadIdx.y * BLOCK_X + threadIdx.x;

  const bool inside = (x < cam.width) && (y < cam.height);
  bool done = !inside;

  const float pixf_x = (float)x + 0.5f;
  const float pixf_y = (float)y + 0.5f;

  const int begin = tile_offsets[tile];
  const int end = tile_offsets[tile + 1];
  const int rounds = (end - begin + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float transmittance = 1.0f;
  float r = 0.0f, g = 0.0f, b = 0.0f;

  int todo = end - begin;
  for (int round = 0; round < rounds; round++, todo -= BLOCK_SIZE) {
    // every pixel of the tile is saturated, so no further Gaussian can
    // contribute to this block
    if (__syncthreads_count(done) == BLOCK_SIZE) break;

    const int fetch = begin + round * BLOCK_SIZE + lane;
    if (fetch < end) {
      const int gid = tile_list[fetch];
      s_xy[lane] = mean2d[gid];
      s_co[lane] = conic_opacity[gid];
      s_color[lane] = color_depth[gid];
    }
    __syncthreads();

    const int count = min(BLOCK_SIZE, todo);
    for (int j = 0; j < count && !done; j++) {
      const float2 xy = s_xy[j];
      const float4 co = s_co[j];
      const float dx = xy.x - pixf_x;
      const float dy = xy.y - pixf_y;

      const float power = -0.5f * fmaf(co.x, dx * dx, co.z * dy * dy) - co.y * dx * dy;
      if (power > 0.0f) continue;

      const float alpha = fminf(0.99f, co.w * __expf(power));
      if (alpha < MIN_OPACITY) continue;

      const float weight = alpha * transmittance;
      const float4 c = s_color[j];
      r = fmaf(c.x, weight, r);
      g = fmaf(c.y, weight, g);
      b = fmaf(c.z, weight, b);

      transmittance *= 1.0f - alpha;
      if (transmittance < MIN_TRANSMITTANCE) done = true;
    }
    __syncthreads();
  }

  if (inside)
    image[(size_t)y * cam.width + x] = make_float4(r, g, b, 1.0f - transmittance);
}

// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of gaussians> <image width> <image height> <repeat>\n",
           argv[0]);
    return 1;
  }

  const int n = atoi(argv[1]);
  const int width = atoi(argv[2]);
  const int height = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  Camera cam;
  setup_camera(width, height, cam);
  MpmParams mpm;
  setup_mpm(mpm);
  const float time = 0.5f;   // the instant of the 4D scene that is rendered

  Scene scene;
  std::vector<float> h_image, h_ref_image;
  std::vector<float> h_mean2d, h_conic, h_color;
  std::vector<int> h_radii;
  std::vector<int> tile_offsets, tile_list;
  try {
    generate_scene(n, scene);
    h_mean2d.resize((size_t)2 * n);
    h_conic.resize((size_t)4 * n);
    h_color.resize((size_t)4 * n);
    h_radii.resize(n);
    h_image.resize(4 * (size_t)width * height);
    h_ref_image.resize(4 * (size_t)width * height);
  } catch (const std::bad_alloc&) {
    printf("Failed to allocate the host buffers for %d gaussians and a %d x %d "
           "image\n", n, width, height);
    return 1;
  }

  const int num_tiles = cam.tiles_x * cam.tiles_y;
  printf("Gaussians: %d, image: %d x %d (%d tiles), MPM grid: %d^3\n",
         n, width, height, num_tiles, MPM_GRID);

  // the host reference evolves its own copy of the scene
  Scene ref_scene = scene;

  float4 *d_mean, *d_scale, *d_quat_l, *d_quat_r, *d_velocity;
  float *d_opacity, *d_sh, *d_affine, *d_defgrad;
  float4 *d_grid;
  CHECK(cudaMalloc((void**)&d_mean, sizeof(float4) * n));
  CHECK(cudaMalloc((void**)&d_scale, sizeof(float4) * n));
  CHECK(cudaMalloc((void**)&d_quat_l, sizeof(float4) * n));
  CHECK(cudaMalloc((void**)&d_quat_r, sizeof(float4) * n));
  CHECK(cudaMalloc((void**)&d_velocity, sizeof(float4) * n));
  CHECK(cudaMalloc((void**)&d_opacity, sizeof(float) * n));
  CHECK(cudaMalloc((void**)&d_sh, sizeof(float) * SH_COEFFS * 3 * (size_t)n));
  CHECK(cudaMalloc((void**)&d_affine, sizeof(float) * 9 * (size_t)n));
  CHECK(cudaMalloc((void**)&d_defgrad, sizeof(float) * 9 * (size_t)n));
  CHECK(cudaMalloc((void**)&d_grid, sizeof(float4) * MPM_CELLS));

  const int num_chunks = (int)scene.chunk_block.size();
  int *d_chunk_start, *d_chunk_block;
  CHECK(cudaMalloc((void**)&d_chunk_start, sizeof(int) * (num_chunks + 1)));
  CHECK(cudaMalloc((void**)&d_chunk_block, sizeof(int) * num_chunks));
  CHECK(cudaMemcpy(d_chunk_start, scene.chunk_start.data(),
                   sizeof(int) * (num_chunks + 1), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_chunk_block, scene.chunk_block.data(),
                   sizeof(int) * num_chunks, cudaMemcpyHostToDevice));

  // the host keeps xyzw and xyz layouts, the device wants 16 byte vectors
  std::vector<float> pack(4 * (size_t)n);
  auto upload_vec4 = [&](const std::vector<float>& src, int comps, float4* dst) {
    for (int i = 0; i < n; i++) {
      pack[4 * (size_t)i + 0] = src[(size_t)comps * i + 0];
      pack[4 * (size_t)i + 1] = src[(size_t)comps * i + 1];
      pack[4 * (size_t)i + 2] = src[(size_t)comps * i + 2];
      pack[4 * (size_t)i + 3] = (comps == 4) ? src[(size_t)comps * i + 3] : 0.0f;
    }
    CHECK(cudaMemcpy(dst, pack.data(), sizeof(float4) * n, cudaMemcpyHostToDevice));
  };

  // the 3x3 tensors are stored component major on the device
  std::vector<float> pack9(9 * (size_t)n);
  auto upload_tensor = [&](const std::vector<float>& src, float* dst) {
    for (int k = 0; k < 9; k++)
      for (int i = 0; i < n; i++) pack9[(size_t)k * n + i] = src[9 * (size_t)i + k];
    CHECK(cudaMemcpy(dst, pack9.data(), sizeof(float) * 9 * (size_t)n,
                     cudaMemcpyHostToDevice));
  };

  upload_vec4(scene.mean, 4, d_mean);
  upload_vec4(scene.scale, 4, d_scale);
  upload_vec4(scene.quat_l, 4, d_quat_l);
  upload_vec4(scene.quat_r, 4, d_quat_r);
  upload_vec4(scene.velocity, 3, d_velocity);
  upload_tensor(scene.affine, d_affine);
  upload_tensor(scene.defgrad, d_defgrad);
  CHECK(cudaMemcpy(d_opacity, scene.opacity.data(), sizeof(float) * n,
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_sh, scene.sh.data(), sizeof(float) * SH_COEFFS * 3 * (size_t)n,
                   cudaMemcpyHostToDevice));

  float2 *d_mean2d;
  float4 *d_conic, *d_color, *d_image;
  int *d_radii, *d_tile_offsets, *d_tile_list;
  CHECK(cudaMalloc((void**)&d_mean2d, sizeof(float2) * n));
  CHECK(cudaMalloc((void**)&d_conic, sizeof(float4) * n));
  CHECK(cudaMalloc((void**)&d_color, sizeof(float4) * n));
  CHECK(cudaMalloc((void**)&d_radii, sizeof(int) * n));
  CHECK(cudaMalloc((void**)&d_image, sizeof(float4) * (size_t)width * height));
  CHECK(cudaMalloc((void**)&d_tile_offsets, sizeof(int) * (num_tiles + 1)));

  const int particle_blocks = (n + P2G_BLOCK - 1) / P2G_BLOCK;
  const int cell_blocks = (MPM_CELLS + GRID_BLOCK - 1) / GRID_BLOCK;
  const int preprocess_blocks = (n + PREPROCESS_BLOCK - 1) / PREPROCESS_BLOCK;

  auto mpm_step = [&]() {
    CHECK(cudaMemset(d_grid, 0, sizeof(float4) * MPM_CELLS));
    mpm_p2g_kernel<<<num_chunks, P2G_BLOCK>>>(
        n, mpm, d_chunk_start, d_chunk_block, d_mean, d_velocity, d_affine,
        d_defgrad, (float*)d_grid);
    mpm_grid_kernel<<<cell_blocks, GRID_BLOCK>>>(mpm, d_grid);
    mpm_g2p_kernel<<<particle_blocks, P2G_BLOCK>>>(
        n, mpm, d_mean, d_velocity, d_affine, d_defgrad, d_grid);
  };

  // --- stage 1: one MPM step, verified against the reference ---------------
  mpm_step();
  CHECK(cudaGetLastError());

  {
    std::vector<float> ref_grid(4 * (size_t)MPM_CELLS);
    reference_p2g(ref_scene, mpm, ref_grid.data());
    reference_grid_update(mpm, ref_grid.data());
    reference_g2p(ref_scene, mpm, ref_grid.data());

    std::vector<float> got(4 * (size_t)n);
    CHECK(cudaMemcpy(got.data(), d_mean, sizeof(float4) * n, cudaMemcpyDeviceToHost));
    int mpm_errors = 0;
    for (int i = 0; i < n && mpm_errors == 0; i++)
      for (int k = 0; k < 3; k++)
        if (!close_enough(got[4 * (size_t)i + k], ref_scene.mean[4 * (size_t)i + k], 1e-4f))
          mpm_errors++;
    CHECK(cudaMemcpy(got.data(), d_velocity, sizeof(float4) * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n && mpm_errors == 0; i++)
      for (int k = 0; k < 3; k++)
        if (!close_enough(got[4 * (size_t)i + k], ref_scene.velocity[3 * (size_t)i + k], 1e-3f))
          mpm_errors++;
    printf("MPM step: %s\n", mpm_errors == 0 ? "PASS" : "FAIL");
  }

  CHECK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) mpm_step();
  CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the MPM step (p2g, grid, g2p): %f (us)\n",
         time_ns * 1e-3 / repeat);

  // --- stage 2: the 4D preprocess, from the undeformed scene ---------------
  upload_vec4(scene.mean, 4, d_mean);

  preprocess_kernel<<<preprocess_blocks, PREPROCESS_BLOCK>>>(
      n, time, cam, d_mean, d_scale, d_quat_l, d_quat_r, d_opacity, d_sh,
      d_mean2d, d_conic, d_color, d_radii);

  std::vector<float> ref_mean2d(2 * (size_t)n), ref_conic(4 * (size_t)n),
      ref_color(4 * (size_t)n);
  {
    std::vector<int> ref_radii(n);
    reference_preprocess(scene, cam, time, ref_mean2d.data(), ref_conic.data(),
                         ref_color.data(), ref_radii.data());

    CHECK(cudaMemcpy(h_mean2d.data(), d_mean2d, sizeof(float2) * n, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_conic.data(), d_conic, sizeof(float4) * n, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_color.data(), d_color, sizeof(float4) * n, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_radii.data(), d_radii, sizeof(int) * n, cudaMemcpyDeviceToHost));

    int pre_errors = 0;
    int visible = 0;
    for (int i = 0; i < n && pre_errors == 0; i++) {
      if (ref_radii[i] > 0) visible++;
      // the extent is the ceiling of a float, so it may land either side of
      // an integer when the host and the device contract differently
      if (abs(h_radii[i] - ref_radii[i]) > 1) pre_errors++;
      if (ref_radii[i] == 0 || h_radii[i] == 0) continue;
      for (int k = 0; k < 2; k++)
        if (!close_enough(h_mean2d[2 * (size_t)i + k], ref_mean2d[2 * (size_t)i + k], 1e-3f))
          pre_errors++;
      for (int k = 0; k < 4; k++) {
        if (!close_enough(h_conic[4 * (size_t)i + k], ref_conic[4 * (size_t)i + k], 1e-3f))
          pre_errors++;
        if (!close_enough(h_color[4 * (size_t)i + k], ref_color[4 * (size_t)i + k], 1e-3f))
          pre_errors++;
      }
    }
    printf("4D preprocess (%d of %d gaussians visible): %s\n", visible, n,
           pre_errors == 0 ? "PASS" : "FAIL");

    // The tile lists are host side setup, shared by the reference and the
    // device rasterizer. Stage 3 is fed the reference splats, so that it is
    // verified on its own rather than against the rounding of stage 2.
    build_tile_lists(cam, n, ref_mean2d.data(), ref_color.data(),
                     ref_radii.data(), tile_offsets, tile_list);
    reference_render(cam, ref_mean2d.data(), ref_conic.data(), ref_color.data(),
                     tile_offsets.data(), tile_list.data(), h_ref_image.data());
  }

  CHECK(cudaDeviceSynchronize());
  start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    preprocess_kernel<<<preprocess_blocks, PREPROCESS_BLOCK>>>(
        n, time, cam, d_mean, d_scale, d_quat_l, d_quat_r, d_opacity, d_sh,
        d_mean2d, d_conic, d_color, d_radii);
  CHECK(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the 4D preprocess kernel: %f (us)\n",
         time_ns * 1e-3 / repeat);

  // --- stage 3: the tile rasterizer ---------------------------------------
  CHECK(cudaMemcpy(d_mean2d, ref_mean2d.data(), sizeof(float2) * n,
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_conic, ref_conic.data(), sizeof(float4) * n,
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_color, ref_color.data(), sizeof(float4) * n,
                   cudaMemcpyHostToDevice));

  const size_t list_size = tile_list.empty() ? 1 : tile_list.size();
  CHECK(cudaMalloc((void**)&d_tile_list, sizeof(int) * list_size));
  CHECK(cudaMemcpy(d_tile_offsets, tile_offsets.data(), sizeof(int) * (num_tiles + 1),
                   cudaMemcpyHostToDevice));
  if (!tile_list.empty())
    CHECK(cudaMemcpy(d_tile_list, tile_list.data(), sizeof(int) * tile_list.size(),
                     cudaMemcpyHostToDevice));

  printf("Gaussian instances after tiling: %zu (%.1f per tile)\n",
         tile_list.size(), (double)tile_list.size() / num_tiles);

  const dim3 render_block(BLOCK_X, BLOCK_Y);
  const dim3 render_grid(cam.tiles_x, cam.tiles_y);

  CHECK(cudaMemset(d_image, 0, sizeof(float4) * (size_t)width * height));
  render_kernel<<<render_grid, render_block>>>(
      cam, d_mean2d, d_conic, d_color, d_tile_offsets, d_tile_list, d_image);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(h_image.data(), d_image, sizeof(float4) * (size_t)width * height,
                   cudaMemcpyDeviceToHost));

  int render_errors = 0;
  for (size_t k = 0; k < h_image.size() && render_errors == 0; k++)
    if (!close_enough(h_image[k], h_ref_image[k], 1e-3f)) render_errors++;
  printf("Rasterizer: %s\n", render_errors == 0 ? "PASS" : "FAIL");

  CHECK(cudaDeviceSynchronize());
  start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    render_kernel<<<render_grid, render_block>>>(
        cam, d_mean2d, d_conic, d_color, d_tile_offsets, d_tile_list, d_image);
  CHECK(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the rasterizer kernel: %f (us)\n",
         time_ns * 1e-3 / repeat);

  CHECK(cudaFree(d_mean)); CHECK(cudaFree(d_scale));
  CHECK(cudaFree(d_quat_l)); CHECK(cudaFree(d_quat_r));
  CHECK(cudaFree(d_velocity)); CHECK(cudaFree(d_opacity));
  CHECK(cudaFree(d_sh)); CHECK(cudaFree(d_affine)); CHECK(cudaFree(d_defgrad));
  CHECK(cudaFree(d_grid)); CHECK(cudaFree(d_mean2d)); CHECK(cudaFree(d_conic));
  CHECK(cudaFree(d_color)); CHECK(cudaFree(d_radii)); CHECK(cudaFree(d_image));
  CHECK(cudaFree(d_tile_offsets)); CHECK(cudaFree(d_tile_list));

  return 0;
}
