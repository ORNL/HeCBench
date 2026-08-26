#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <new>
#include <optional>
#include <vector>
#include <sycl/sycl.hpp>
#include "reference.h"

#define P2G_BLOCK 128
#define GRID_BLOCK 256
#define PREPROCESS_BLOCK 128

// 16 byte vectors, so that a work item loads a pose or a splat with one
// 128-bit access
struct alignas(16) Float4 { float x, y, z, w; };
struct alignas(8) Float2 { float x, y; };

using AtomicGlobal =
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>;
using AtomicLocal =
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>;

static inline void quad_weights(float fx, float& w0, float& w1, float& w2)
{
  const float a = 1.5f - fx;
  const float b = fx - 1.0f;
  const float c = fx - 0.5f;
  w0 = 0.5f * a * a;
  w1 = 0.75f - b * b;
  w2 = 0.5f * c * c;
}

// ---------------------------------------------------------------------------
// stage 1: MLS-MPM
// ---------------------------------------------------------------------------

// One work group scatters one chunk of particles, all of which belong to the
// same MPM_BLOCK^3 cell block, so their stencil footprint fits a MPM_TILE^3
// tile in local memory. A particle that has drifted out of its block since
// the binning still lands correctly through the global fallback.
static void mpm_p2g(sycl::nd_item<1>& item, float* tile, int n, const MpmParams p,
                    const int* __restrict chunk_start,
                    const int* __restrict chunk_block,
                    const Float4* __restrict mean,
                    const Float4* __restrict velocity,
                    const float* __restrict affine_in,
                    const float* __restrict defgrad,
                    float* __restrict grid)
{
  auto group = item.get_group();
  const int lid = item.get_local_id(0);

  for (int k = lid; k < MPM_TILE_CELLS * 4; k += P2G_BLOCK) tile[k] = 0.0f;

  const int chunk = item.get_group(0);
  const int begin = chunk_start[chunk];
  const int end = chunk_start[chunk + 1];

  const int gb = chunk_block[chunk];
  const int obz = (gb % MPM_BLOCKS_PER_DIM) * MPM_BLOCK - 1;
  const int oby = ((gb / MPM_BLOCKS_PER_DIM) % MPM_BLOCKS_PER_DIM) * MPM_BLOCK - 1;
  const int obx = (gb / (MPM_BLOCKS_PER_DIM * MPM_BLOCKS_PER_DIM)) * MPM_BLOCK - 1;

  sycl::group_barrier(group);

  for (int i = begin + lid; i < end; i += P2G_BLOCK) {
    const Float4 x = mean[i];
    const Float4 v = velocity[i];

    float C[9], F0[9];
    #pragma unroll
    for (int k = 0; k < 9; k++) C[k] = affine_in[(size_t)k * n + i];
    #pragma unroll
    for (int k = 0; k < 9; k++) F0[k] = defgrad[(size_t)k * n + i];

    float F[9];
    #pragma unroll
    for (int a = 0; a < 3; a++)
      #pragma unroll
      for (int b = 0; b < 3; b++) {
        float acc = F0[a * 3 + b];
        #pragma unroll
        for (int k = 0; k < 3; k++)
          acc = sycl::fma(p.dt * C[a * 3 + k], F0[k * 3 + b], acc);
        F[a * 3 + b] = acc;
      }

    const float det =
        F[0] * (F[4] * F[8] - F[5] * F[7]) -
        F[1] * (F[3] * F[8] - F[5] * F[6]) +
        F[2] * (F[3] * F[7] - F[4] * F[6]);
    const float safe_J =
        (sycl::fabs(det) < 1e-6f) ? ((det < 0.0f) ? -1e-6f : 1e-6f) : det;
    const float inv_J = 1.0f / safe_J;

    const float cof[9] = {
       (F[4] * F[8] - F[5] * F[7]), -(F[3] * F[8] - F[5] * F[6]),  (F[3] * F[7] - F[4] * F[6]),
      -(F[1] * F[8] - F[2] * F[7]),  (F[0] * F[8] - F[2] * F[6]), -(F[0] * F[7] - F[1] * F[6]),
       (F[1] * F[5] - F[2] * F[4]), -(F[0] * F[5] - F[2] * F[3]),  (F[0] * F[4] - F[1] * F[3]) };

    const float coeff =
        (p.lambda * sycl::native::log(sycl::fabs(safe_J)) - p.mu) * inv_J;
    float P[9];
    #pragma unroll
    for (int k = 0; k < 9; k++) P[k] = sycl::fma(coeff, cof[k], p.mu * F[k]);

    const float s = -p.dt * p.particle_volume * 4.0f * p.inv_dx * p.inv_dx;
    float affine[9];
    #pragma unroll
    for (int a = 0; a < 3; a++)
      #pragma unroll
      for (int b = 0; b < 3; b++) {
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 3; k++)
          acc = sycl::fma(P[a * 3 + k], F[b * 3 + k], acc);
        affine[a * 3 + b] = sycl::fma(s, acc, p.particle_mass * C[a * 3 + b]);
      }

    // F is only used for the stress above; it is deliberately not persisted
    // here. The single per-step deformation-gradient update is applied once, in
    // mpm_g2p, from the original F0 (this matches the host reference; writing F
    // back here would advance the gradient twice per step).

    const float gx = x.x * p.inv_dx, gy = x.y * p.inv_dx, gz = x.z * p.inv_dx;
    const int bx = (int)sycl::floor(gx - 0.5f);
    const int by = (int)sycl::floor(gy - 0.5f);
    const int bz = (int)sycl::floor(gz - 0.5f);
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

          float val[4];
          #pragma unroll
          for (int k = 0; k < 3; k++) {
            float impulse = affine[k * 3 + 0] * dpx;
            impulse = sycl::fma(affine[k * 3 + 1], dpy, impulse);
            impulse = sycl::fma(affine[k * 3 + 2], dpz, impulse);
            val[k] = w * (mv[k] + impulse);
          }
          val[3] = w * p.particle_mass;

          const int lx = ix - obx, ly = iy - oby, lz = iz - obz;
          if ((unsigned)lx < MPM_TILE && (unsigned)ly < MPM_TILE &&
              (unsigned)lz < MPM_TILE) {
            float* cell = tile + 4 * ((lx * MPM_TILE + ly) * MPM_TILE + lz);
            #pragma unroll
            for (int k = 0; k < 4; k++) AtomicLocal(cell[k]).fetch_add(val[k]);
          } else {
            float* cell =
                grid + 4 * ((size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz);
            #pragma unroll
            for (int k = 0; k < 4; k++) AtomicGlobal(cell[k]).fetch_add(val[k]);
          }
        }
      }
    }
  }

  sycl::group_barrier(group);

  for (int c = lid; c < MPM_TILE_CELLS; c += P2G_BLOCK) {
    const int lz = c % MPM_TILE;
    const int ly = (c / MPM_TILE) % MPM_TILE;
    const int lx = c / (MPM_TILE * MPM_TILE);
    const int ix = obx + lx, iy = oby + ly, iz = obz + lz;
    if ((unsigned)ix >= MPM_GRID || (unsigned)iy >= MPM_GRID ||
        (unsigned)iz >= MPM_GRID)
      continue;

    const float* acc = tile + 4 * c;
    if (acc[0] == 0.0f && acc[1] == 0.0f && acc[2] == 0.0f && acc[3] == 0.0f)
      continue;

    float* dst = grid + 4 * ((size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz);
    #pragma unroll
    for (int k = 0; k < 4; k++) AtomicGlobal(dst[k]).fetch_add(acc[k]);
  }
}

static void mpm_grid_update(sycl::nd_item<1>& item, const MpmParams p,
                            Float4* __restrict grid)
{
  const int cell = item.get_global_id(0);
  if (cell >= MPM_CELLS) return;

  Float4 g = grid[cell];
  if (g.w <= 0.0f) {
    grid[cell] = Float4{ 0.0f, 0.0f, 0.0f, g.w };
    return;
  }

  const float inv_mass = 1.0f / g.w;
  float vx = g.x * inv_mass;
  float vy = sycl::fma(p.dt, p.gravity, g.y * inv_mass);
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

  grid[cell] = Float4{ vx, vy, vz, g.w };
}

static void mpm_g2p(sycl::nd_item<1>& item, int n, const MpmParams p,
                    Float4* __restrict mean,
                    Float4* __restrict velocity,
                    float* __restrict affine_out,
                    float* __restrict defgrad,
                    const Float4* __restrict grid)
{
  const int i = item.get_global_id(0);
  if (i >= n) return;

  const Float4 x = mean[i];
  const float gx = x.x * p.inv_dx, gy = x.y * p.inv_dx, gz = x.z * p.inv_dx;
  const int bx = (int)sycl::floor(gx - 0.5f);
  const int by = (int)sycl::floor(gy - 0.5f);
  const int bz = (int)sycl::floor(gz - 0.5f);
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
        const Float4 g = grid[(size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz];
        const float gv[3] = { g.x, g.y, g.z };
        const float dpos[3] = { dpx, dpy, ((float)c - fz) * p.dx };

        #pragma unroll
        for (int k = 0; k < 3; k++) {
          nv[k] = sycl::fma(w, gv[k], nv[k]);
          const float wg = 4.0f * p.inv_dx * p.inv_dx * w * gv[k];
          #pragma unroll
          for (int l = 0; l < 3; l++)
            nC[k * 3 + l] = sycl::fma(wg, dpos[l], nC[k * 3 + l]);
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
        acc = sycl::fma(p.dt * nC[a * 3 + k], F0[k * 3 + b], acc);
      defgrad[(size_t)(a * 3 + b) * n + i] = acc;
    }

  #pragma unroll
  for (int k = 0; k < 9; k++) affine_out[(size_t)k * n + i] = nC[k];

  velocity[i] = Float4{ nv[0], nv[1], nv[2], 0.0f };
  mean[i] = Float4{ sycl::fma(p.dt, nv[0], x.x), sycl::fma(p.dt, nv[1], x.y),
                    sycl::fma(p.dt, nv[2], x.z), x.w };
}

// ---------------------------------------------------------------------------
// stage 2: condition the 4D Gaussian on t, project, shade
// ---------------------------------------------------------------------------

static void preprocess(sycl::nd_item<1>& item, int n, float time, const Camera cam,
                       const Float4* __restrict mean4,
                       const Float4* __restrict scale4,
                       const Float4* __restrict quat_l,
                       const Float4* __restrict quat_r,
                       const float* __restrict opacity_in,
                       const float* __restrict sh,
                       Float2* __restrict mean2d,
                       Float4* __restrict conic_opacity,
                       Float4* __restrict color_depth,
                       int* __restrict radii)
{
  const int i = item.get_global_id(0);
  if (i >= n) return;

  radii[i] = 0;

  const Float4 mu = mean4[i];
  const Float4 sc = scale4[i];
  const Float4 ql = quat_l[i];
  const Float4 qr = quat_r[i];

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
      for (int k = 0; k < 4; k++) acc = sycl::fma(L[a * 4 + k], R[k * 4 + b], acc);
      M[a * 4 + b] = acc;
    }

  // the quaternion basis is ordered (t, x, y, z)
  const float s2[4] = { sc.w * sc.w, sc.x * sc.x, sc.y * sc.y, sc.z * sc.z };

  float st[3], sxyz[6];
  #pragma unroll
  for (int a = 0; a < 3; a++) {
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++)
      acc = sycl::fma(M[(a + 1) * 4 + k] * s2[k], M[k], acc);
    st[a] = acc;
  }
  float sigma_tt = 0.0f;
  #pragma unroll
  for (int k = 0; k < 4; k++) sigma_tt = sycl::fma(M[k] * s2[k], M[k], sigma_tt);

  int idx = 0;
  #pragma unroll
  for (int a = 0; a < 3; a++)
    #pragma unroll
    for (int b = a; b < 3; b++) {
      float acc = 0.0f;
      #pragma unroll
      for (int k = 0; k < 4; k++)
        acc = sycl::fma(M[(a + 1) * 4 + k] * s2[k], M[(b + 1) * 4 + k], acc);
      sxyz[idx++] = acc;
    }

  const float inv_tt = 1.0f / sigma_tt;
  const float dt = time - mu.w;

  const float alpha = opacity_in[i] * sycl::native::exp(-0.5f * dt * dt * inv_tt);
  if (alpha < MIN_OPACITY) return;

  const float mean3[3] = { sycl::fma(dt * inv_tt, st[0], mu.x),
                           sycl::fma(dt * inv_tt, st[1], mu.y),
                           sycl::fma(dt * inv_tt, st[2], mu.z) };
  const float cov3[6] = { sxyz[0] - st[0] * st[0] * inv_tt,
                          sxyz[1] - st[0] * st[1] * inv_tt,
                          sxyz[2] - st[0] * st[2] * inv_tt,
                          sxyz[3] - st[1] * st[1] * inv_tt,
                          sxyz[4] - st[1] * st[2] * inv_tt,
                          sxyz[5] - st[2] * st[2] * inv_tt };

  const float px = mean3[0] - cam.cam_pos[0];
  const float py = mean3[1] - cam.cam_pos[1];
  const float pz = mean3[2] - cam.cam_pos[2];
  float t[3];
  #pragma unroll
  for (int a = 0; a < 3; a++)
    t[a] = sycl::fma(cam.view[a * 3 + 0], px,
                     sycl::fma(cam.view[a * 3 + 1], py, cam.view[a * 3 + 2] * pz));

  if (t[2] < cam.near_plane) return;

  const float inv_z = 1.0f / t[2];
  const float inv_z2 = inv_z * inv_z;

  const float lim_x = 1.3f * (0.5f * cam.width) / cam.focal_x;
  const float lim_y = 1.3f * (0.5f * cam.height) / cam.focal_y;
  const float tx = t[2] * sycl::fmin(lim_x, sycl::fmax(-lim_x, t[0] * inv_z));
  const float ty = t[2] * sycl::fmin(lim_y, sycl::fmax(-lim_y, t[1] * inv_z));

  const float J[6] = { cam.focal_x * inv_z, 0.0f, -cam.focal_x * tx * inv_z2,
                       0.0f, cam.focal_y * inv_z, -cam.focal_y * ty * inv_z2 };

  float T[6];
  #pragma unroll
  for (int a = 0; a < 2; a++)
    #pragma unroll
    for (int b = 0; b < 3; b++)
      T[a * 3 + b] = sycl::fma(J[a * 3 + 0], cam.view[0 * 3 + b],
                               sycl::fma(J[a * 3 + 1], cam.view[1 * 3 + b],
                                         J[a * 3 + 2] * cam.view[2 * 3 + b]));

  const float S[9] = { cov3[0], cov3[1], cov3[2],
                       cov3[1], cov3[3], cov3[4],
                       cov3[2], cov3[4], cov3[5] };
  float TS[6];
  #pragma unroll
  for (int a = 0; a < 2; a++)
    #pragma unroll
    for (int b = 0; b < 3; b++)
      TS[a * 3 + b] = sycl::fma(T[a * 3 + 0], S[0 * 3 + b],
                                sycl::fma(T[a * 3 + 1], S[1 * 3 + b],
                                          T[a * 3 + 2] * S[2 * 3 + b]));

  float ca = 0.3f, cb = 0.0f, cc = 0.3f;
  #pragma unroll
  for (int k = 0; k < 3; k++) {
    ca = sycl::fma(TS[0 * 3 + k], T[0 * 3 + k], ca);
    cb = sycl::fma(TS[0 * 3 + k], T[1 * 3 + k], cb);
    cc = sycl::fma(TS[1 * 3 + k], T[1 * 3 + k], cc);
  }

  const float det = sycl::fma(ca, cc, -cb * cb);
  if (det <= 0.0f) return;
  const float inv_det = 1.0f / det;

  const float mid = 0.5f * (ca + cc);
  const float disc = sycl::sqrt(sycl::fmax(0.1f, sycl::fma(mid, mid, -det)));
  const float radius = sycl::ceil(3.0f * sycl::sqrt(sycl::fmax(mid + disc, mid - disc)));

  const float mx = sycl::fma(cam.focal_x * inv_z, t[0], cam.center_x);
  const float my = sycl::fma(cam.focal_y * inv_z, t[1], cam.center_y);

  if (mx + radius < 0.0f || mx - radius > (float)cam.width ||
      my + radius < 0.0f || my - radius > (float)cam.height)
    return;

  const float inv_len = sycl::rsqrt(sycl::fma(px, px, sycl::fma(py, py, pz * pz)));
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
      rgb[ch] = sycl::fma(w, sh[((size_t)c * 3 + ch) * n + i], rgb[ch]);
  }

  radii[i] = (int)radius;
  mean2d[i] = Float2{ mx, my };
  conic_opacity[i] = Float4{ cc * inv_det, -cb * inv_det, ca * inv_det, alpha };
  color_depth[i] = Float4{ sycl::fmax(rgb[0] + 0.5f, 0.0f),
                           sycl::fmax(rgb[1] + 0.5f, 0.0f),
                           sycl::fmax(rgb[2] + 0.5f, 0.0f), t[2] };
}

// ---------------------------------------------------------------------------
// stage 3: tile rasterizer
// ---------------------------------------------------------------------------

static void render(sycl::nd_item<2>& item, Float2* s_xy, Float4* s_co,
                   Float4* s_color, const Camera cam,
                   const Float2* __restrict mean2d,
                   const Float4* __restrict conic_opacity,
                   const Float4* __restrict color_depth,
                   const int* __restrict tile_offsets,
                   const int* __restrict tile_list,
                   Float4* __restrict image)
{
  const int tile = item.get_group(0) * cam.tiles_x + item.get_group(1);
  const int x = item.get_global_id(1);
  const int y = item.get_global_id(0);
  const int lane = item.get_local_id(0) * BLOCK_X + item.get_local_id(1);

  const bool valid = (x < cam.width) && (y < cam.height);
  bool done = !valid;

  const float pixf_x = (float)x + 0.5f;
  const float pixf_y = (float)y + 0.5f;

  const int begin = tile_offsets[tile];
  const int end = tile_offsets[tile + 1];
  const int rounds = (end - begin + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float transmittance = 1.0f;
  float r = 0.0f, g = 0.0f, b = 0.0f;

  int todo = end - begin;
  auto group = item.get_group();

  for (int round = 0; round < rounds; round++, todo -= BLOCK_SIZE) {
    // every pixel of the tile is saturated, so no further Gaussian can
    // contribute to this work group
    if (sycl::all_of_group(group, done)) break;

    const int fetch = begin + round * BLOCK_SIZE + lane;
    if (fetch < end) {
      const int gid = tile_list[fetch];
      s_xy[lane] = mean2d[gid];
      s_co[lane] = conic_opacity[gid];
      s_color[lane] = color_depth[gid];
    }
    sycl::group_barrier(group);

    const int count = sycl::min(BLOCK_SIZE, todo);
    for (int j = 0; j < count && !done; j++) {
      const Float2 xy = s_xy[j];
      const Float4 co = s_co[j];
      const float dx = xy.x - pixf_x;
      const float dy = xy.y - pixf_y;

      const float power =
          -0.5f * sycl::fma(co.x, dx * dx, co.z * dy * dy) - co.y * dx * dy;
      if (power > 0.0f) continue;

      const float alpha = sycl::fmin(0.99f, co.w * sycl::native::exp(power));
      if (alpha < MIN_OPACITY) continue;

      const float weight = alpha * transmittance;
      const Float4 c = s_color[j];
      r = sycl::fma(c.x, weight, r);
      g = sycl::fma(c.y, weight, g);
      b = sycl::fma(c.z, weight, b);

      transmittance *= 1.0f - alpha;
      if (transmittance < MIN_TRANSMITTANCE) done = true;
    }
    sycl::group_barrier(group);
  }

  if (valid)
    image[(size_t)y * cam.width + x] = Float4{ r, g, b, 1.0f - transmittance };
}

// ---------------------------------------------------------------------------

static int run(int argc, char* argv[])
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

  if (n <= 0 || width <= 0 || height <= 0 || repeat <= 0) {
    printf("Error: number of gaussians, image width, image height, and repeat "
           "must all be positive integers (got n=%d, width=%d, height=%d, "
           "repeat=%d)\n", n, width, height, repeat);
    return 1;
  }

  Camera cam;
  setup_camera(width, height, cam);
  MpmParams mpm;
  setup_mpm(mpm);
  const float time = 0.5f;

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

  Scene ref_scene = scene;

  // held by value in an optional: a default constructed queue would run the
  // default selector, which throws outside the handler when no device exists
  std::optional<sycl::queue> queue;
  try {
#ifdef USE_GPU
    queue.emplace(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    queue.emplace(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  } catch (const sycl::exception& e) {
    printf("Failed to select a SYCL device: %s\n", e.what());
    return 1;
  }
  sycl::queue& q = *queue;

  const int num_chunks = (int)scene.chunk_block.size();

  Float4* d_mean = sycl::malloc_device<Float4>(n, q);
  Float4* d_scale = sycl::malloc_device<Float4>(n, q);
  Float4* d_quat_l = sycl::malloc_device<Float4>(n, q);
  Float4* d_quat_r = sycl::malloc_device<Float4>(n, q);
  Float4* d_velocity = sycl::malloc_device<Float4>(n, q);
  float* d_opacity = sycl::malloc_device<float>(n, q);
  float* d_sh = sycl::malloc_device<float>((size_t)SH_COEFFS * 3 * n, q);
  float* d_affine = sycl::malloc_device<float>((size_t)9 * n, q);
  float* d_defgrad = sycl::malloc_device<float>((size_t)9 * n, q);
  Float4* d_grid = sycl::malloc_device<Float4>(MPM_CELLS, q);
  int* d_chunk_start = sycl::malloc_device<int>(num_chunks + 1, q);
  int* d_chunk_block = sycl::malloc_device<int>(num_chunks, q);
  Float2* d_mean2d = sycl::malloc_device<Float2>(n, q);
  Float4* d_conic = sycl::malloc_device<Float4>(n, q);
  Float4* d_color = sycl::malloc_device<Float4>(n, q);
  int* d_radii = sycl::malloc_device<int>(n, q);
  Float4* d_image = sycl::malloc_device<Float4>((size_t)width * height, q);
  int* d_tile_offsets = sycl::malloc_device<int>(num_tiles + 1, q);

  if (d_mean == nullptr || d_scale == nullptr || d_quat_l == nullptr ||
      d_quat_r == nullptr || d_velocity == nullptr || d_opacity == nullptr ||
      d_sh == nullptr || d_affine == nullptr || d_defgrad == nullptr ||
      d_grid == nullptr || d_chunk_start == nullptr || d_chunk_block == nullptr ||
      d_mean2d == nullptr || d_conic == nullptr || d_color == nullptr ||
      d_radii == nullptr || d_image == nullptr || d_tile_offsets == nullptr) {
    printf("Failed to allocate the device buffers for %d gaussians and a "
           "%d x %d image\n", n, width, height);
    return 1;
  }

  std::vector<Float4> pack(n);
  auto upload_vec4 = [&](const std::vector<float>& src, int comps, Float4* dst) {
    for (int i = 0; i < n; i++)
      pack[i] = Float4{ src[(size_t)comps * i + 0], src[(size_t)comps * i + 1],
                        src[(size_t)comps * i + 2],
                        (comps == 4) ? src[(size_t)comps * i + 3] : 0.0f };
    q.memcpy(dst, pack.data(), sizeof(Float4) * n).wait();
  };

  std::vector<float> pack9((size_t)9 * n);
  auto upload_tensor = [&](const std::vector<float>& src, float* dst) {
    for (int k = 0; k < 9; k++)
      for (int i = 0; i < n; i++) pack9[(size_t)k * n + i] = src[9 * (size_t)i + k];
    q.memcpy(dst, pack9.data(), sizeof(float) * 9 * (size_t)n).wait();
  };

  upload_vec4(scene.mean, 4, d_mean);
  upload_vec4(scene.scale, 4, d_scale);
  upload_vec4(scene.quat_l, 4, d_quat_l);
  upload_vec4(scene.quat_r, 4, d_quat_r);
  upload_vec4(scene.velocity, 3, d_velocity);
  upload_tensor(scene.affine, d_affine);
  upload_tensor(scene.defgrad, d_defgrad);
  q.memcpy(d_opacity, scene.opacity.data(), sizeof(float) * n);
  q.memcpy(d_sh, scene.sh.data(), sizeof(float) * SH_COEFFS * 3 * (size_t)n);
  q.memcpy(d_chunk_start, scene.chunk_start.data(), sizeof(int) * (num_chunks + 1));
  q.memcpy(d_chunk_block, scene.chunk_block.data(), sizeof(int) * num_chunks);
  //q.wait_and_throw();

  const int cell_groups = (MPM_CELLS + GRID_BLOCK - 1) / GRID_BLOCK;
  const int particle_groups = (n + P2G_BLOCK - 1) / P2G_BLOCK;

  auto mpm_step = [&]() {
    q.memset(d_grid, 0, sizeof(Float4) * MPM_CELLS);
    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> tile(sycl::range<1>(MPM_TILE_CELLS * 4), cgh);
      cgh.parallel_for<class p2g>(
          sycl::nd_range<1>(sycl::range<1>((size_t)num_chunks * P2G_BLOCK),
                            sycl::range<1>(P2G_BLOCK)),
          [=](sycl::nd_item<1> item) {
            mpm_p2g(item, tile.get_multi_ptr<sycl::access::decorated::no>().get(),
                    n, mpm, d_chunk_start, d_chunk_block, d_mean, d_velocity,
                    d_affine, d_defgrad, (float*)d_grid);
          });
    });
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class grid_update>(
          sycl::nd_range<1>(sycl::range<1>((size_t)cell_groups * GRID_BLOCK),
                            sycl::range<1>(GRID_BLOCK)),
          [=](sycl::nd_item<1> item) { mpm_grid_update(item, mpm, d_grid); });
    });
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class g2p>(
          sycl::nd_range<1>(sycl::range<1>((size_t)particle_groups * P2G_BLOCK),
                            sycl::range<1>(P2G_BLOCK)),
          [=](sycl::nd_item<1> item) {
            mpm_g2p(item, n, mpm, d_mean, d_velocity, d_affine, d_defgrad, d_grid);
          });
    });
  };

  // --- stage 1 ------------------------------------------------------------
  mpm_step();
  //q.wait_and_throw();

  {
    std::vector<float> ref_grid(4 * (size_t)MPM_CELLS);
    reference_p2g(ref_scene, mpm, ref_grid.data());
    reference_grid_update(mpm, ref_grid.data());
    reference_g2p(ref_scene, mpm, ref_grid.data());

    std::vector<Float4> got(n);
    q.memcpy(got.data(), d_mean, sizeof(Float4) * n).wait();
    int mpm_errors = 0;
    for (int i = 0; i < n && mpm_errors == 0; i++) {
      const float* r = &ref_scene.mean[4 * (size_t)i];
      if (!close_enough(got[i].x, r[0], 1e-4f) ||
          !close_enough(got[i].y, r[1], 1e-4f) ||
          !close_enough(got[i].z, r[2], 1e-4f))
        mpm_errors++;
    }
    q.memcpy(got.data(), d_velocity, sizeof(Float4) * n).wait();
    for (int i = 0; i < n && mpm_errors == 0; i++) {
      const float* r = &ref_scene.velocity[3 * (size_t)i];
      if (!close_enough(got[i].x, r[0], 1e-3f) ||
          !close_enough(got[i].y, r[1], 1e-3f) ||
          !close_enough(got[i].z, r[2], 1e-3f))
        mpm_errors++;
    }
    printf("MPM step: %s\n", mpm_errors == 0 ? "PASS" : "FAIL");
  }

  q.wait_and_throw();
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) mpm_step();
  q.wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the MPM step (p2g, grid, g2p): %f (us)\n",
         time_ns * 1e-3 / repeat);

  // --- stage 2 ------------------------------------------------------------
  upload_vec4(scene.mean, 4, d_mean);

  auto preprocess_launch = [&]() {
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class preprocess_4d>(
          sycl::nd_range<1>(
              sycl::range<1>((size_t)((n + PREPROCESS_BLOCK - 1) / PREPROCESS_BLOCK) *
                             PREPROCESS_BLOCK),
              sycl::range<1>(PREPROCESS_BLOCK)),
          [=](sycl::nd_item<1> item) {
            preprocess(item, n, time, cam, d_mean, d_scale, d_quat_l, d_quat_r,
                       d_opacity, d_sh, d_mean2d, d_conic, d_color, d_radii);
          });
    });
  };

  preprocess_launch();
  //q.wait_and_throw();

  std::vector<float> ref_mean2d(2 * (size_t)n), ref_conic(4 * (size_t)n),
      ref_color(4 * (size_t)n);
  {
    std::vector<int> ref_radii(n);
    reference_preprocess(scene, cam, time, ref_mean2d.data(), ref_conic.data(),
                         ref_color.data(), ref_radii.data());

    q.memcpy(h_mean2d.data(), d_mean2d, sizeof(Float2) * n);
    q.memcpy(h_conic.data(), d_conic, sizeof(Float4) * n);
    q.memcpy(h_color.data(), d_color, sizeof(Float4) * n);
    q.memcpy(h_radii.data(), d_radii, sizeof(int) * n);
    //q.wait_and_throw();

    int pre_errors = 0;
    int visible = 0;
    for (int i = 0; i < n && pre_errors == 0; i++) {
      if (ref_radii[i] > 0) visible++;
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

    build_tile_lists(cam, n, ref_mean2d.data(), ref_color.data(),
                     ref_radii.data(), tile_offsets, tile_list);
    reference_render(cam, ref_mean2d.data(), ref_conic.data(), ref_color.data(),
                     tile_offsets.data(), tile_list.data(), h_ref_image.data());
  }

  q.wait_and_throw();
  start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) preprocess_launch();
  q.wait_and_throw();
  end = std::chrono::steady_clock::now();
  time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the 4D preprocess kernel: %f (us)\n",
         time_ns * 1e-3 / repeat);

  // --- stage 3 ------------------------------------------------------------
  q.memcpy(d_mean2d, ref_mean2d.data(), sizeof(Float2) * n);
  q.memcpy(d_conic, ref_conic.data(), sizeof(Float4) * n);
  q.memcpy(d_color, ref_color.data(), sizeof(Float4) * n);

  const size_t list_size = tile_list.empty() ? 1 : tile_list.size();
  int* d_tile_list = sycl::malloc_device<int>(list_size, q);
  if (d_tile_list == nullptr) {
    printf("Failed to allocate the device buffer for %zu gaussian instances\n",
           list_size);
    return 1;
  }
  q.memcpy(d_tile_offsets, tile_offsets.data(), sizeof(int) * (num_tiles + 1));
  if (!tile_list.empty())
    q.memcpy(d_tile_list, tile_list.data(), sizeof(int) * tile_list.size());
  //q.wait_and_throw();

  printf("Gaussian instances after tiling: %zu (%.1f per tile)\n",
         tile_list.size(), (double)tile_list.size() / num_tiles);

  const sycl::range<2> global_range((size_t)cam.tiles_y * BLOCK_Y,
                                    (size_t)cam.tiles_x * BLOCK_X);
  const sycl::range<2> local_range(BLOCK_Y, BLOCK_X);

  auto render_launch = [&]() {
    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<Float2, 1> s_xy(sycl::range<1>(BLOCK_SIZE), cgh);
      sycl::local_accessor<Float4, 1> s_co(sycl::range<1>(BLOCK_SIZE), cgh);
      sycl::local_accessor<Float4, 1> s_color(sycl::range<1>(BLOCK_SIZE), cgh);
      cgh.parallel_for<class rasterize>(
          sycl::nd_range<2>(global_range, local_range),
          [=](sycl::nd_item<2> item) {
            render(item, s_xy.get_multi_ptr<sycl::access::decorated::no>().get(),
                   s_co.get_multi_ptr<sycl::access::decorated::no>().get(),
                   s_color.get_multi_ptr<sycl::access::decorated::no>().get(),
                   cam, d_mean2d, d_conic, d_color, d_tile_offsets, d_tile_list,
                   d_image);
          });
    });
  };

  q.memset(d_image, 0, sizeof(Float4) * (size_t)width * height).wait();
  render_launch();
  q.memcpy(h_image.data(), d_image, sizeof(Float4) * (size_t)width * height).wait();

  {
    int render_errors = 0;
    for (size_t k = 0; k < h_image.size() && render_errors == 0; k++)
      if (!close_enough(h_image[k], h_ref_image[k], 1e-3f)) render_errors++;
    printf("Rasterizer: %s\n", render_errors == 0 ? "PASS" : "FAIL");
  }

  q.wait_and_throw();
  start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) render_launch();
  q.wait_and_throw();
  end = std::chrono::steady_clock::now();
  time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the rasterizer kernel: %f (us)\n",
         time_ns * 1e-3 / repeat);

  sycl::free(d_mean, q); sycl::free(d_scale, q); sycl::free(d_quat_l, q);
  sycl::free(d_quat_r, q); sycl::free(d_velocity, q); sycl::free(d_opacity, q);
  sycl::free(d_sh, q); sycl::free(d_affine, q); sycl::free(d_defgrad, q);
  sycl::free(d_grid, q); sycl::free(d_chunk_start, q); sycl::free(d_chunk_block, q);
  sycl::free(d_mean2d, q); sycl::free(d_conic, q); sycl::free(d_color, q);
  sycl::free(d_radii, q); sycl::free(d_image, q); sycl::free(d_tile_offsets, q);
  sycl::free(d_tile_list, q);

  return 0;
}

int main(int argc, char* argv[])
{
  // allocation and submission can throw too, so the whole run is guarded
  try {
    return run(argc, argv);
  } catch (const sycl::exception& e) {
    printf("SYCL error: %s\n", e.what());
    return 1;
  }
}
