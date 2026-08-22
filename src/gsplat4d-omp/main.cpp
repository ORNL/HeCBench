#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <new>
#include <vector>
#include <omp.h>
#include "reference.h"

#define P2G_BLOCK 128
#define PREPROCESS_BLOCK 128

#pragma omp declare target
static inline void quad_weights(float fx, float& w0, float& w1, float& w2)
{
  const float a = 1.5f - fx;
  const float b = fx - 1.0f;
  const float c = fx - 0.5f;
  w0 = 0.5f * a * a;
  w1 = 0.75f - b * b;
  w2 = 0.5f * c * c;
}
#pragma omp end declare target

// ---------------------------------------------------------------------------
// stage 1: MLS-MPM
//
// One team scatters one chunk of particles, all of which belong to the same
// MPM_BLOCK^3 cell block, so the whole stencil footprint fits the team local
// tile below and the scatter only reaches the global grid at the flush. A
// particle that has drifted out of its block since the binning still lands
// correctly through the global fallback.
// ---------------------------------------------------------------------------

static void mpm_p2g(int n, int num_chunks,
                    const int* chunk_start, const int* chunk_block,
                    const float* mean, const float* velocity,
                    const float* affine, const float* defgrad, float* grid,
                    float dt, float dx, float inv_dx, float p_vol, float p_mass,
                    float mu_lame, float lambda_lame)
{
  #pragma omp target teams num_teams(num_chunks) thread_limit(P2G_BLOCK)
  {
    float tile[MPM_TILE_CELLS * 4];
    #pragma omp parallel num_threads(P2G_BLOCK)
    {
      const int team = omp_get_team_num();
      const int nteams = omp_get_num_teams();
      const int lid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();

      // The runtime may create fewer than num_chunks teams, so each team walks
      // a strided range of chunks rather than assuming a one-to-one mapping.
      // Every thread of a team runs the same iteration count, so the team
      // barriers below stay aligned.
      for (int chunk = team; chunk < num_chunks; chunk += nteams) {
        for (int k = lid; k < MPM_TILE_CELLS * 4; k += nthreads) tile[k] = 0.0f;

        const int begin = chunk_start[chunk];
        const int end = chunk_start[chunk + 1];
        const int gb = chunk_block[chunk];
        const int obz = (gb % MPM_BLOCKS_PER_DIM) * MPM_BLOCK - 1;
        const int oby = ((gb / MPM_BLOCKS_PER_DIM) % MPM_BLOCKS_PER_DIM) * MPM_BLOCK - 1;
        const int obx = (gb / (MPM_BLOCKS_PER_DIM * MPM_BLOCKS_PER_DIM)) * MPM_BLOCK - 1;

        #pragma omp barrier

        for (int i = begin + lid; i < end; i += nthreads) {
          float C[9], F0[9], F[9];
          for (int k = 0; k < 9; k++) C[k] = affine[(size_t)k * n + i];
          for (int k = 0; k < 9; k++) F0[k] = defgrad[(size_t)k * n + i];

          for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) {
              float acc = F0[a * 3 + b];
              for (int k = 0; k < 3; k++) acc += dt * C[a * 3 + k] * F0[k * 3 + b];
              F[a * 3 + b] = acc;
            }

          const float det =
              F[0] * (F[4] * F[8] - F[5] * F[7]) -
              F[1] * (F[3] * F[8] - F[5] * F[6]) +
              F[2] * (F[3] * F[7] - F[4] * F[6]);
          const float safe_J =
              (fabsf(det) < 1e-6f) ? ((det < 0.0f) ? -1e-6f : 1e-6f) : det;
          const float inv_J = 1.0f / safe_J;

          const float cof[9] = {
             (F[4] * F[8] - F[5] * F[7]), -(F[3] * F[8] - F[5] * F[6]),  (F[3] * F[7] - F[4] * F[6]),
            -(F[1] * F[8] - F[2] * F[7]),  (F[0] * F[8] - F[2] * F[6]), -(F[0] * F[7] - F[1] * F[6]),
             (F[1] * F[5] - F[2] * F[4]), -(F[0] * F[5] - F[2] * F[3]),  (F[0] * F[4] - F[1] * F[3]) };

          const float coeff = (lambda_lame * logf(fabsf(safe_J)) - mu_lame) * inv_J;
          float P[9];
          for (int k = 0; k < 9; k++) P[k] = mu_lame * F[k] + coeff * cof[k];

          const float s = -dt * p_vol * 4.0f * inv_dx * inv_dx;
          float aff[9];
          for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) {
              float acc = 0.0f;
              for (int k = 0; k < 3; k++) acc += P[a * 3 + k] * F[b * 3 + k];
              aff[a * 3 + b] = s * acc + p_mass * C[a * 3 + b];
            }

          // F is only used for the stress above; it is deliberately not
          // persisted here. The single per-step deformation-gradient update is
          // applied once, in mpm_g2p, from the original F0 (matching the host
          // reference; writing F back here would advance the gradient twice).

          const float gx = mean[4 * (size_t)i + 0] * inv_dx;
          const float gy = mean[4 * (size_t)i + 1] * inv_dx;
          const float gz = mean[4 * (size_t)i + 2] * inv_dx;
          const int bx = (int)floorf(gx - 0.5f);
          const int by = (int)floorf(gy - 0.5f);
          const int bz = (int)floorf(gz - 0.5f);
          const float fx = gx - bx, fy = gy - by, fz = gz - bz;

          float wx[3], wy[3], wz[3];
          quad_weights(fx, wx[0], wx[1], wx[2]);
          quad_weights(fy, wy[0], wy[1], wy[2]);
          quad_weights(fz, wz[0], wz[1], wz[2]);

          const float mv[3] = { p_mass * velocity[4 * (size_t)i + 0],
                                p_mass * velocity[4 * (size_t)i + 1],
                                p_mass * velocity[4 * (size_t)i + 2] };

          for (int a = 0; a < 3; a++) {
            const int ix = bx + a;
            if (ix < 0 || ix >= MPM_GRID) continue;
            const float dpx = ((float)a - fx) * dx;
            for (int b = 0; b < 3; b++) {
              const int iy = by + b;
              if (iy < 0 || iy >= MPM_GRID) continue;
              const float dpy = ((float)b - fy) * dx;
              const float wxy = wx[a] * wy[b];
              for (int c = 0; c < 3; c++) {
                const int iz = bz + c;
                if (iz < 0 || iz >= MPM_GRID) continue;
                const float dpz = ((float)c - fz) * dx;
                const float w = wxy * wz[c];

                float val[4];
                for (int k = 0; k < 3; k++)
                  val[k] = w * (mv[k] + aff[k * 3 + 0] * dpx +
                                aff[k * 3 + 1] * dpy + aff[k * 3 + 2] * dpz);
                val[3] = w * p_mass;

                const int lx = ix - obx, ly = iy - oby, lz = iz - obz;
                if ((unsigned)lx < MPM_TILE && (unsigned)ly < MPM_TILE &&
                    (unsigned)lz < MPM_TILE) {
                  const int o = 4 * ((lx * MPM_TILE + ly) * MPM_TILE + lz);
                  for (int k = 0; k < 4; k++) {
                    #pragma omp atomic update
                    tile[o + k] += val[k];
                  }
                } else {
                  const size_t o = 4 * ((size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz);
                  for (int k = 0; k < 4; k++) {
                    #pragma omp atomic update
                    grid[o + k] += val[k];
                  }
                }
              }
            }
          }
        }

        #pragma omp barrier

        for (int c = lid; c < MPM_TILE_CELLS; c += nthreads) {
          const int lz = c % MPM_TILE;
          const int ly = (c / MPM_TILE) % MPM_TILE;
          const int lx = c / (MPM_TILE * MPM_TILE);
          const int ix = obx + lx, iy = oby + ly, iz = obz + lz;
          if ((unsigned)ix >= MPM_GRID || (unsigned)iy >= MPM_GRID ||
              (unsigned)iz >= MPM_GRID)
            continue;
          if (tile[4 * c + 0] == 0.0f && tile[4 * c + 1] == 0.0f &&
              tile[4 * c + 2] == 0.0f && tile[4 * c + 3] == 0.0f)
            continue;
          const size_t o = 4 * ((size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz);
          for (int k = 0; k < 4; k++) {
            #pragma omp atomic update
            grid[o + k] += tile[4 * c + k];
          }
        }

        // finish the flush before the next chunk re-zeros the shared tile
        #pragma omp barrier
      }
    }
  }
}

static void mpm_grid_update(float* grid, float dt, float gravity, int boundary)
{
  #pragma omp target teams distribute parallel for
  for (int cell = 0; cell < MPM_CELLS; cell++) {
    const size_t o = 4 * (size_t)cell;
    const float mass = grid[o + 3];
    if (mass <= 0.0f) {
      grid[o + 0] = 0.0f;
      grid[o + 1] = 0.0f;
      grid[o + 2] = 0.0f;
      continue;
    }
    const float inv_mass = 1.0f / mass;
    float vx = grid[o + 0] * inv_mass;
    float vy = grid[o + 1] * inv_mass + dt * gravity;
    float vz = grid[o + 2] * inv_mass;

    const int iz = cell % MPM_GRID;
    const int iy = (cell / MPM_GRID) % MPM_GRID;
    const int ix = cell / (MPM_GRID * MPM_GRID);

    if (ix < boundary && vx < 0.0f) vx = 0.0f;
    if (ix >= MPM_GRID - boundary && vx > 0.0f) vx = 0.0f;
    if (iy < boundary && vy < 0.0f) vy = 0.0f;
    if (iy >= MPM_GRID - boundary && vy > 0.0f) vy = 0.0f;
    if (iz < boundary && vz < 0.0f) vz = 0.0f;
    if (iz >= MPM_GRID - boundary && vz > 0.0f) vz = 0.0f;

    grid[o + 0] = vx;
    grid[o + 1] = vy;
    grid[o + 2] = vz;
  }
}

static void mpm_g2p(int n, float* mean, float* velocity, float* affine,
                    float* defgrad, const float* grid,
                    float dt, float dx, float inv_dx)
{
  #pragma omp target teams distribute parallel for thread_limit(P2G_BLOCK)
  for (int i = 0; i < n; i++) {
    const float gx = mean[4 * (size_t)i + 0] * inv_dx;
    const float gy = mean[4 * (size_t)i + 1] * inv_dx;
    const float gz = mean[4 * (size_t)i + 2] * inv_dx;
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

    for (int a = 0; a < 3; a++) {
      const int ix = bx + a;
      if (ix < 0 || ix >= MPM_GRID) continue;
      const float dpx = ((float)a - fx) * dx;
      for (int b = 0; b < 3; b++) {
        const int iy = by + b;
        if (iy < 0 || iy >= MPM_GRID) continue;
        const float dpy = ((float)b - fy) * dx;
        const float wxy = wx[a] * wy[b];
        for (int c = 0; c < 3; c++) {
          const int iz = bz + c;
          if (iz < 0 || iz >= MPM_GRID) continue;
          const float w = wxy * wz[c];
          const size_t o = 4 * ((size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz);
          const float gv[3] = { grid[o + 0], grid[o + 1], grid[o + 2] };
          const float dpos[3] = { dpx, dpy, ((float)c - fz) * dx };
          for (int k = 0; k < 3; k++) {
            nv[k] += w * gv[k];
            const float wg = 4.0f * inv_dx * inv_dx * w * gv[k];
            for (int l = 0; l < 3; l++) nC[k * 3 + l] += wg * dpos[l];
          }
        }
      }
    }

    float F0[9];
    for (int k = 0; k < 9; k++) F0[k] = defgrad[(size_t)k * n + i];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        float acc = F0[a * 3 + b];
        for (int k = 0; k < 3; k++) acc += dt * nC[a * 3 + k] * F0[k * 3 + b];
        defgrad[(size_t)(a * 3 + b) * n + i] = acc;
      }
    for (int k = 0; k < 9; k++) affine[(size_t)k * n + i] = nC[k];
    for (int k = 0; k < 3; k++) {
      velocity[4 * (size_t)i + k] = nv[k];
      mean[4 * (size_t)i + k] += dt * nv[k];
    }
  }
}

static void mpm_step(int n, int num_chunks, const int* chunk_start,
                     const int* chunk_block, float* mean, float* velocity,
                     float* affine, float* defgrad, float* grid,
                     const MpmParams p)
{
  #pragma omp target teams distribute parallel for
  for (size_t k = 0; k < 4 * (size_t)MPM_CELLS; k++) grid[k] = 0.0f;

  mpm_p2g(n, num_chunks, chunk_start, chunk_block, mean, velocity, affine,
          defgrad, grid, p.dt, p.dx, p.inv_dx, p.particle_volume,
          p.particle_mass, p.mu, p.lambda);
  mpm_grid_update(grid, p.dt, p.gravity, p.boundary);
  mpm_g2p(n, mean, velocity, affine, defgrad, grid, p.dt, p.dx, p.inv_dx);
}

// ---------------------------------------------------------------------------
// stage 2: condition the 4D Gaussian on t, project, shade
// ---------------------------------------------------------------------------

static void preprocess(int n, float time, const float* mean, const float* scale,
                       const float* quat_l, const float* quat_r,
                       const float* opacity, const float* sh,
                       const float* view, const float* cam_pos,
                       float focal_x, float focal_y, float center_x,
                       float center_y, float near_plane, int width, int height,
                       float* mean2d, float* conic, float* color, int* radii)
{
  #pragma omp target teams distribute parallel for thread_limit(PREPROCESS_BLOCK)
  for (int i = 0; i < n; i++) {
    radii[i] = 0;

    const float lw = quat_l[4 * (size_t)i + 0], lx = quat_l[4 * (size_t)i + 1];
    const float ly = quat_l[4 * (size_t)i + 2], lz = quat_l[4 * (size_t)i + 3];
    const float rw = quat_r[4 * (size_t)i + 0], rx = quat_r[4 * (size_t)i + 1];
    const float ry = quat_r[4 * (size_t)i + 2], rz = quat_r[4 * (size_t)i + 3];

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
    for (int a = 0; a < 4; a++)
      for (int b = 0; b < 4; b++) {
        float acc = 0.0f;
        for (int k = 0; k < 4; k++) acc += L[a * 4 + k] * R[k * 4 + b];
        M[a * 4 + b] = acc;
      }

    // the quaternion basis is ordered (t, x, y, z)
    const float sx = scale[4 * (size_t)i + 0], sy = scale[4 * (size_t)i + 1];
    const float sz = scale[4 * (size_t)i + 2], stime = scale[4 * (size_t)i + 3];
    const float s2[4] = { stime * stime, sx * sx, sy * sy, sz * sz };

    float st[3], sxyz[6];
    for (int a = 0; a < 3; a++) {
      float acc = 0.0f;
      for (int k = 0; k < 4; k++) acc += M[(a + 1) * 4 + k] * s2[k] * M[k];
      st[a] = acc;
    }
    float sigma_tt = 0.0f;
    for (int k = 0; k < 4; k++) sigma_tt += M[k] * s2[k] * M[k];

    int idx = 0;
    for (int a = 0; a < 3; a++)
      for (int b = a; b < 3; b++) {
        float acc = 0.0f;
        for (int k = 0; k < 4; k++)
          acc += M[(a + 1) * 4 + k] * s2[k] * M[(b + 1) * 4 + k];
        sxyz[idx++] = acc;
      }

    const float inv_tt = 1.0f / sigma_tt;
    const float tdiff = time - mean[4 * (size_t)i + 3];

    const float alpha = opacity[i] * expf(-0.5f * tdiff * tdiff * inv_tt);
    if (alpha < MIN_OPACITY) continue;

    const float mean3[3] = { mean[4 * (size_t)i + 0] + tdiff * inv_tt * st[0],
                             mean[4 * (size_t)i + 1] + tdiff * inv_tt * st[1],
                             mean[4 * (size_t)i + 2] + tdiff * inv_tt * st[2] };
    const float cov3[6] = { sxyz[0] - st[0] * st[0] * inv_tt,
                            sxyz[1] - st[0] * st[1] * inv_tt,
                            sxyz[2] - st[0] * st[2] * inv_tt,
                            sxyz[3] - st[1] * st[1] * inv_tt,
                            sxyz[4] - st[1] * st[2] * inv_tt,
                            sxyz[5] - st[2] * st[2] * inv_tt };

    const float px = mean3[0] - cam_pos[0];
    const float py = mean3[1] - cam_pos[1];
    const float pz = mean3[2] - cam_pos[2];
    float t[3];
    for (int a = 0; a < 3; a++)
      t[a] = view[a * 3 + 0] * px + view[a * 3 + 1] * py + view[a * 3 + 2] * pz;

    if (t[2] < near_plane) continue;

    const float inv_z = 1.0f / t[2];
    const float inv_z2 = inv_z * inv_z;

    const float lim_x = 1.3f * (0.5f * width) / focal_x;
    const float lim_y = 1.3f * (0.5f * height) / focal_y;
    const float tx = t[2] * fminf(lim_x, fmaxf(-lim_x, t[0] * inv_z));
    const float ty = t[2] * fminf(lim_y, fmaxf(-lim_y, t[1] * inv_z));

    const float J[6] = { focal_x * inv_z, 0.0f, -focal_x * tx * inv_z2,
                         0.0f, focal_y * inv_z, -focal_y * ty * inv_z2 };

    float T[6];
    for (int a = 0; a < 2; a++)
      for (int b = 0; b < 3; b++)
        T[a * 3 + b] = J[a * 3 + 0] * view[0 * 3 + b] +
                       J[a * 3 + 1] * view[1 * 3 + b] +
                       J[a * 3 + 2] * view[2 * 3 + b];

    const float S[9] = { cov3[0], cov3[1], cov3[2],
                         cov3[1], cov3[3], cov3[4],
                         cov3[2], cov3[4], cov3[5] };
    float TS[6];
    for (int a = 0; a < 2; a++)
      for (int b = 0; b < 3; b++)
        TS[a * 3 + b] = T[a * 3 + 0] * S[0 * 3 + b] + T[a * 3 + 1] * S[1 * 3 + b] +
                        T[a * 3 + 2] * S[2 * 3 + b];

    float ca = 0.3f, cb = 0.0f, cc = 0.3f;
    for (int k = 0; k < 3; k++) {
      ca += TS[0 * 3 + k] * T[0 * 3 + k];
      cb += TS[0 * 3 + k] * T[1 * 3 + k];
      cc += TS[1 * 3 + k] * T[1 * 3 + k];
    }

    const float det = ca * cc - cb * cb;
    if (det <= 0.0f) continue;
    const float inv_det = 1.0f / det;

    const float mid = 0.5f * (ca + cc);
    const float disc = sqrtf(fmaxf(0.1f, mid * mid - det));
    const float radius = ceilf(3.0f * sqrtf(fmaxf(mid + disc, mid - disc)));

    const float mx = focal_x * t[0] * inv_z + center_x;
    const float my = focal_y * t[1] * inv_z + center_y;

    if (mx + radius < 0.0f || mx - radius > (float)width ||
        my + radius < 0.0f || my - radius > (float)height)
      continue;

    const float inv_len = 1.0f / sqrtf(px * px + py * py + pz * pz);
    const float ddx = px * inv_len, ddy = py * inv_len, ddz = pz * inv_len;

    float basis[SH_COEFFS];
    basis[0] = SH_C0;
    basis[1] = -SH_C1 * ddy;
    basis[2] = SH_C1 * ddz;
    basis[3] = -SH_C1 * ddx;

    const float xx = ddx * ddx, yy = ddy * ddy, zz = ddz * ddz;
    const float xy = ddx * ddy, yz = ddy * ddz, xz = ddx * ddz;
    basis[4] = SH_C2_0 * xy;
    basis[5] = SH_C2_1 * yz;
    basis[6] = SH_C2_2 * (2.0f * zz - xx - yy);
    basis[7] = SH_C2_3 * xz;
    basis[8] = SH_C2_4 * (xx - yy);
    basis[9]  = SH_C3_0 * ddy * (3.0f * xx - yy);
    basis[10] = SH_C3_1 * xy * ddz;
    basis[11] = SH_C3_2 * ddy * (4.0f * zz - xx - yy);
    basis[12] = SH_C3_3 * ddz * (2.0f * zz - 3.0f * xx - 3.0f * yy);
    basis[13] = SH_C3_4 * ddx * (4.0f * zz - xx - yy);
    basis[14] = SH_C3_5 * ddz * (xx - yy);
    basis[15] = SH_C3_6 * ddx * (xx - 3.0f * yy);

    float rgb[3] = { 0.0f, 0.0f, 0.0f };
    for (int c = 0; c < SH_COEFFS; c++)
      for (int ch = 0; ch < 3; ch++)
        rgb[ch] += basis[c] * sh[((size_t)c * 3 + ch) * n + i];

    radii[i] = (int)radius;
    mean2d[2 * (size_t)i + 0] = mx;
    mean2d[2 * (size_t)i + 1] = my;
    conic[4 * (size_t)i + 0] = cc * inv_det;
    conic[4 * (size_t)i + 1] = -cb * inv_det;
    conic[4 * (size_t)i + 2] = ca * inv_det;
    conic[4 * (size_t)i + 3] = alpha;
    color[4 * (size_t)i + 0] = fmaxf(rgb[0] + 0.5f, 0.0f);
    color[4 * (size_t)i + 1] = fmaxf(rgb[1] + 0.5f, 0.0f);
    color[4 * (size_t)i + 2] = fmaxf(rgb[2] + 0.5f, 0.0f);
    color[4 * (size_t)i + 3] = t[2];
  }
}

// ---------------------------------------------------------------------------
// stage 3: tile rasterizer
//
// One team owns one 16x16 tile and stages a batch of Gaussians in the team
// local arrays, so each Gaussian is read once per tile rather than once per
// pixel, and the team leaves as soon as all of its pixels are saturated.
// ---------------------------------------------------------------------------

static void render(int num_tiles, int tiles_x, int width, int height,
                   const float* mean2d, const float* conic, const float* color,
                   const int* offsets, const int* list, float* image)
{
  #pragma omp target teams num_teams(num_tiles) thread_limit(BLOCK_SIZE)
  {
    float s_xy[2 * BLOCK_SIZE];
    float s_co[4 * BLOCK_SIZE];
    float s_color[4 * BLOCK_SIZE];
    // per-pixel accumulation state, kept in team-shared memory so that all
    // team barriers live outside the per-pixel loop
    float p_trans[BLOCK_SIZE];
    float p_r[BLOCK_SIZE];
    float p_g[BLOCK_SIZE];
    float p_b[BLOCK_SIZE];
    #pragma omp parallel num_threads(BLOCK_SIZE)
    {
      const int team = omp_get_team_num();
      const int nteams = omp_get_num_teams();
      const int lane = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();

      // The runtime may create fewer than num_tiles teams and fewer than
      // BLOCK_SIZE threads, so each team strides over tiles and the round count
      // is derived from the thread count. Every thread of a team executes the
      // same number of rounds, so the barriers below stay aligned regardless of
      // how BLOCK_SIZE divides the thread count.
      for (int tile = team; tile < num_tiles; tile += nteams) {
        const int tx = tile % tiles_x, ty = tile / tiles_x;
        const int begin = offsets[tile];
        const int end = offsets[tile + 1];
        const int total = end - begin;
        const int rounds = (total + nthreads - 1) / nthreads;

        for (int pixel = lane; pixel < BLOCK_SIZE; pixel += nthreads) {
          p_trans[pixel] = 1.0f;
          p_r[pixel] = 0.0f;
          p_g[pixel] = 0.0f;
          p_b[pixel] = 0.0f;
        }
        #pragma omp barrier

        for (int round = 0; round < rounds; round++) {
          const int fetch = begin + round * nthreads + lane;
          if (fetch < end) {
            const int gid = list[fetch];
            s_xy[2 * lane + 0] = mean2d[2 * (size_t)gid + 0];
            s_xy[2 * lane + 1] = mean2d[2 * (size_t)gid + 1];
            for (int k = 0; k < 4; k++) {
              s_co[4 * lane + k] = conic[4 * (size_t)gid + k];
              s_color[4 * lane + k] = color[4 * (size_t)gid + k];
            }
          }
          #pragma omp barrier

          const int remaining = total - round * nthreads;
          const int count = (nthreads < remaining) ? nthreads : remaining;

          for (int pixel = lane; pixel < BLOCK_SIZE; pixel += nthreads) {
            const int x = tx * BLOCK_X + pixel % BLOCK_X;
            const int y = ty * BLOCK_Y + pixel / BLOCK_X;
            if (x >= width || y >= height) continue;

            float transmittance = p_trans[pixel];
            if (transmittance < MIN_TRANSMITTANCE) continue;

            const float pixf_x = (float)x + 0.5f;
            const float pixf_y = (float)y + 0.5f;
            float r = p_r[pixel], g = p_g[pixel], b = p_b[pixel];

            for (int j = 0; j < count; j++) {
              const float ddx = s_xy[2 * j + 0] - pixf_x;
              const float ddy = s_xy[2 * j + 1] - pixf_y;
              const float power = -0.5f * (s_co[4 * j + 0] * ddx * ddx +
                                           s_co[4 * j + 2] * ddy * ddy) -
                                  s_co[4 * j + 1] * ddx * ddy;
              if (power > 0.0f) continue;

              const float a = s_co[4 * j + 3] * expf(power);
              const float alpha = (a < 0.99f) ? a : 0.99f;
              if (alpha < MIN_OPACITY) continue;

              const float weight = alpha * transmittance;
              r += s_color[4 * j + 0] * weight;
              g += s_color[4 * j + 1] * weight;
              b += s_color[4 * j + 2] * weight;

              transmittance *= 1.0f - alpha;
              if (transmittance < MIN_TRANSMITTANCE) break;
            }

            p_trans[pixel] = transmittance;
            p_r[pixel] = r;
            p_g[pixel] = g;
            p_b[pixel] = b;
          }
          // finish shading against this batch before it is overwritten
          #pragma omp barrier
        }

        for (int pixel = lane; pixel < BLOCK_SIZE; pixel += nthreads) {
          const int x = tx * BLOCK_X + pixel % BLOCK_X;
          const int y = ty * BLOCK_Y + pixel / BLOCK_X;
          if (x >= width || y >= height) continue;
          const size_t o = 4 * ((size_t)y * width + x);
          image[o + 0] = p_r[pixel];
          image[o + 1] = p_g[pixel];
          image[o + 2] = p_b[pixel];
          image[o + 3] = 1.0f - p_trans[pixel];
        }
        // finish writes/reads of the shared state before the next tile reuses it
        #pragma omp barrier
      }
    }
  }
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
  const float render_time = 0.5f;

  Scene scene;
  std::vector<float> h_image, h_ref_image;
  std::vector<float> h_mean, h_velocity, h_affine, h_defgrad, h_grid;
  std::vector<float> h_mean2d, h_conic, h_color;
  std::vector<int> h_radii;
  std::vector<int> tile_offsets, tile_list;
  try {
    generate_scene(n, scene);
    h_mean.resize((size_t)4 * n);
    h_velocity.resize((size_t)4 * n);
    h_affine.resize((size_t)9 * n);
    h_defgrad.resize((size_t)9 * n);
    h_grid.resize(4 * (size_t)MPM_CELLS);
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

  // the device sees the same layouts as the other backends: xyzw vectors for
  // the poses, component major for the 3x3 tensors
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < 4; k++) h_mean[4 * (size_t)i + k] = scene.mean[4 * (size_t)i + k];
    for (int k = 0; k < 3; k++) h_velocity[4 * (size_t)i + k] = scene.velocity[3 * (size_t)i + k];
    h_velocity[4 * (size_t)i + 3] = 0.0f;
    for (int k = 0; k < 9; k++) {
      h_affine[(size_t)k * n + i] = scene.affine[9 * (size_t)i + k];
      h_defgrad[(size_t)k * n + i] = scene.defgrad[9 * (size_t)i + k];
    }
  }

  float* mean = h_mean.data();
  float* velocity = h_velocity.data();
  float* affine = h_affine.data();
  float* defgrad = h_defgrad.data();
  float* grid = h_grid.data();
  const float* scale = scene.scale.data();
  const float* quat_l = scene.quat_l.data();
  const float* quat_r = scene.quat_r.data();
  const float* opacity = scene.opacity.data();
  const float* sh = scene.sh.data();
  float* mean2d = h_mean2d.data();
  float* conic = h_conic.data();
  float* color = h_color.data();
  int* radii = h_radii.data();
  float* image = h_image.data();

  const int num_chunks = (int)scene.chunk_block.size();
  const int* chunk_start = scene.chunk_start.data();
  const int* chunk_block = scene.chunk_block.data();

  const size_t sh_size = (size_t)SH_COEFFS * 3 * n;
  const size_t image_size = 4 * (size_t)width * height;

  float view[9], cam_pos[3];
  for (int k = 0; k < 9; k++) view[k] = cam.view[k];
  for (int k = 0; k < 3; k++) cam_pos[k] = cam.cam_pos[k];

  #pragma omp target data \
    map(tofrom: mean[0:4*(size_t)n], velocity[0:4*(size_t)n], \
                affine[0:9*(size_t)n], defgrad[0:9*(size_t)n]) \
    map(alloc: grid[0:4*(size_t)MPM_CELLS]) \
    map(to: scale[0:4*(size_t)n], quat_l[0:4*(size_t)n], quat_r[0:4*(size_t)n], \
            opacity[0:n], sh[0:sh_size], view[0:9], cam_pos[0:3], \
            chunk_start[0:num_chunks+1], chunk_block[0:num_chunks]) \
    map(tofrom: mean2d[0:2*(size_t)n], conic[0:4*(size_t)n], \
                color[0:4*(size_t)n], radii[0:n])
  {

  // --- stage 1: one MPM step, verified against the reference ---------------
  mpm_step(n, num_chunks, chunk_start, chunk_block, mean, velocity, affine,
           defgrad, grid, mpm);

  {
    std::vector<float> ref_grid(4 * (size_t)MPM_CELLS);
    reference_p2g(ref_scene, mpm, ref_grid.data());
    reference_grid_update(mpm, ref_grid.data());
    reference_g2p(ref_scene, mpm, ref_grid.data());

    #pragma omp target update from(mean[0:4*(size_t)n], velocity[0:4*(size_t)n])

    int mpm_errors = 0;
    for (int i = 0; i < n && mpm_errors == 0; i++)
      for (int k = 0; k < 3; k++)
        if (!close_enough(mean[4 * (size_t)i + k], ref_scene.mean[4 * (size_t)i + k], 1e-4f))
          mpm_errors++;
    for (int i = 0; i < n && mpm_errors == 0; i++)
      for (int k = 0; k < 3; k++)
        if (!close_enough(velocity[4 * (size_t)i + k], ref_scene.velocity[3 * (size_t)i + k], 1e-3f))
          mpm_errors++;
    printf("MPM step: %s\n", mpm_errors == 0 ? "PASS" : "FAIL");
  }

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    mpm_step(n, num_chunks, chunk_start, chunk_block, mean, velocity, affine,
             defgrad, grid, mpm);
  auto end = std::chrono::steady_clock::now();
  auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the MPM step (p2g, grid, g2p): %f (us)\n",
         time_ns * 1e-3 / repeat);

  // --- stage 2: the 4D preprocess, from the undeformed scene ---------------
  for (int i = 0; i < n; i++)
    for (int k = 0; k < 4; k++)
      mean[4 * (size_t)i + k] = scene.mean[4 * (size_t)i + k];
  #pragma omp target update to(mean[0:4*(size_t)n])

  preprocess(n, render_time, mean, scale, quat_l, quat_r, opacity, sh, view,
             cam_pos, cam.focal_x, cam.focal_y, cam.center_x, cam.center_y,
             cam.near_plane, width, height, mean2d, conic, color, radii);

  std::vector<float> ref_mean2d(2 * (size_t)n), ref_conic(4 * (size_t)n),
      ref_color(4 * (size_t)n);
  {
    std::vector<int> ref_radii(n);
    reference_preprocess(scene, cam, render_time, ref_mean2d.data(),
                         ref_conic.data(), ref_color.data(), ref_radii.data());

    #pragma omp target update from(mean2d[0:2*(size_t)n], conic[0:4*(size_t)n], \
                                   color[0:4*(size_t)n], radii[0:n])

    int pre_errors = 0;
    int visible = 0;
    for (int i = 0; i < n && pre_errors == 0; i++) {
      if (ref_radii[i] > 0) visible++;
      if (abs(radii[i] - ref_radii[i]) > 1) pre_errors++;
      if (ref_radii[i] == 0 || radii[i] == 0) continue;
      for (int k = 0; k < 2; k++)
        if (!close_enough(mean2d[2 * (size_t)i + k], ref_mean2d[2 * (size_t)i + k], 1e-3f))
          pre_errors++;
      for (int k = 0; k < 4; k++) {
        if (!close_enough(conic[4 * (size_t)i + k], ref_conic[4 * (size_t)i + k], 1e-3f))
          pre_errors++;
        if (!close_enough(color[4 * (size_t)i + k], ref_color[4 * (size_t)i + k], 1e-3f))
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

  start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    preprocess(n, render_time, mean, scale, quat_l, quat_r, opacity, sh, view,
               cam_pos, cam.focal_x, cam.focal_y, cam.center_x, cam.center_y,
               cam.near_plane, width, height, mean2d, conic, color, radii);
  end = std::chrono::steady_clock::now();
  time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the 4D preprocess kernel: %f (us)\n",
         time_ns * 1e-3 / repeat);

  // --- stage 3: the tile rasterizer ---------------------------------------
  for (size_t k = 0; k < 2 * (size_t)n; k++) mean2d[k] = ref_mean2d[k];
  for (size_t k = 0; k < 4 * (size_t)n; k++) conic[k] = ref_conic[k];
  for (size_t k = 0; k < 4 * (size_t)n; k++) color[k] = ref_color[k];
  #pragma omp target update to(mean2d[0:2*(size_t)n], conic[0:4*(size_t)n], \
                               color[0:4*(size_t)n])

  if (tile_list.empty()) tile_list.push_back(0);
  const int list_size = (int)tile_list.size();
  const int* offsets = tile_offsets.data();
  const int* list = tile_list.data();

  printf("Gaussian instances after tiling: %d (%.1f per tile)\n",
         list_size, (double)list_size / num_tiles);

  #pragma omp target data map(to: offsets[0:num_tiles+1], list[0:list_size]) \
                          map(from: image[0:image_size])
  {
  render(num_tiles, cam.tiles_x, width, height, mean2d, conic, color, offsets,
         list, image);

  #pragma omp target update from(image[0:image_size])

  {
    int render_errors = 0;
    for (size_t k = 0; k < image_size && render_errors == 0; k++)
      if (!close_enough(image[k], h_ref_image[k], 1e-3f)) render_errors++;
    printf("Rasterizer: %s\n", render_errors == 0 ? "PASS" : "FAIL");
  }

  start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    render(num_tiles, cam.tiles_x, width, height, mean2d, conic, color, offsets,
           list, image);
  end = std::chrono::steady_clock::now();
  time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the rasterizer kernel: %f (us)\n",
         time_ns * 1e-3 / repeat);
  }

  }

  return 0;
}
