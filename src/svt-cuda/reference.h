#ifndef REFERENCE_H
#define REFERENCE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <vector>

// Sparse-volume ray casting benchmark capturing the hierarchical traversal
// and irregular memory-access patterns of NanoVDB-based TSDF renderers
// (https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb).
//
// The volume is a two-level sparse grid whose blocks are traversed with a 3-D
// DDA at the block level and a fine-grained DDA at the voxel level inside each
// active block.  The TSDF is procedurally generated from a union of spheres so
// the benchmark is self-contained and has no external data dependencies.
//
// Reference application: slim-vdb
// (https://github.com/umfieldrobotics/slim-vdb), which ray-casts NanoVDB
// grids for semantic 3-D mapping in field robotics.

// Tile size for GPU kernel launch
#define BLOCK_X 16
#define BLOCK_Y 16

// Leaf (block) dimensions: 8^3 voxels per leaf
#define LEAF_LOG2   3
#define LEAF_DIM    (1 << LEAF_LOG2)
#define LEAF_MASK   (LEAF_DIM - 1)
#define LEAF_VOXELS (LEAF_DIM * LEAF_DIM * LEAF_DIM)

// Coarse grid of blocks: 64^3
#define GRID_DIM    64
#define GRID_BLOCKS (GRID_DIM * GRID_DIM * GRID_DIM)

// Total voxel resolution: 512^3
#define VOXEL_DIM   (GRID_DIM * LEAF_DIM)
#define VOXEL_SIZE  (1.0f / VOXEL_DIM)
#define BLOCK_W     (LEAF_DIM * VOXEL_SIZE)

// TSDF truncation distance (in world coordinates)
#define TRUNC_DIST  (4.0f * VOXEL_SIZE)

#define MAX_SPHERES 32
#define BG_COLOR    0.1f

struct Camera {
  float eye[3];
  float right[3];
  float up[3];
  float forward[3];
  float fov_scale;
  float inv_width, inv_height;
  int   width, height;
};

struct Sphere {
  float cx, cy, cz, r;
};

struct SparseVolume {
  int   block_offset[GRID_BLOCKS];
  std::vector<float> leaf_data;
  int   num_active;
};

// ---------------------------------------------------------------------------
// deterministic RNG (identical to the one in gsplat4d)
// ---------------------------------------------------------------------------

static inline unsigned rng_next(unsigned& state)
{
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static inline float rng_uniform(unsigned& state)
{
  return (rng_next(state) >> 8) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------------
// scene generation
// ---------------------------------------------------------------------------

static void generate_spheres(Sphere* spheres, int num_spheres)
{
  unsigned state = 42u;
  for (int i = 0; i < num_spheres; i++) {
    spheres[i].cx = 0.15f + 0.7f * rng_uniform(state);
    spheres[i].cy = 0.15f + 0.7f * rng_uniform(state);
    spheres[i].cz = 0.15f + 0.7f * rng_uniform(state);
    spheres[i].r  = 0.05f + 0.12f * rng_uniform(state);
  }
}

static inline float sphere_sdf(float x, float y, float z, const Sphere& s)
{
  float dx = x - s.cx, dy = y - s.cy, dz = z - s.cz;
  return sqrtf(dx * dx + dy * dy + dz * dz) - s.r;
}

static inline float scene_sdf(float x, float y, float z,
                               const Sphere* spheres, int num_spheres)
{
  float d = FLT_MAX;
  for (int i = 0; i < num_spheres; i++)
    d = std::min(d, sphere_sdf(x, y, z, spheres[i]));
  return d;
}

static void build_volume(SparseVolume& vol,
                          const Sphere* spheres, int num_spheres)
{
  vol.num_active = 0;

  const float block_diag = sqrtf(3.0f) * 0.5f * BLOCK_W;

  for (int bx = 0; bx < GRID_DIM; bx++)
    for (int by = 0; by < GRID_DIM; by++)
      for (int bz = 0; bz < GRID_DIM; bz++) {
        int bi = (bx * GRID_DIM + by) * GRID_DIM + bz;

        float cx = (bx + 0.5f) * BLOCK_W;
        float cy = (by + 0.5f) * BLOCK_W;
        float cz = (bz + 0.5f) * BLOCK_W;

        bool active = false;
        for (int s = 0; s < num_spheres && !active; s++)
          if (fabsf(sphere_sdf(cx, cy, cz, spheres[s])) < TRUNC_DIST + block_diag)
            active = true;

        vol.block_offset[bi] = active ? vol.num_active++ : -1;
      }

  vol.leaf_data.resize((size_t)vol.num_active * LEAF_VOXELS);

  for (int bx = 0; bx < GRID_DIM; bx++)
    for (int by = 0; by < GRID_DIM; by++)
      for (int bz = 0; bz < GRID_DIM; bz++) {
        int bi  = (bx * GRID_DIM + by) * GRID_DIM + bz;
        int off = vol.block_offset[bi];
        if (off < 0) continue;

        float bx0 = bx * BLOCK_W, by0 = by * BLOCK_W, bz0 = bz * BLOCK_W;

        for (int lx = 0; lx < LEAF_DIM; lx++)
          for (int ly = 0; ly < LEAF_DIM; ly++)
            for (int lz = 0; lz < LEAF_DIM; lz++) {
              float x = bx0 + (lx + 0.5f) * VOXEL_SIZE;
              float y = by0 + (ly + 0.5f) * VOXEL_SIZE;
              float z = bz0 + (lz + 0.5f) * VOXEL_SIZE;

              float d = scene_sdf(x, y, z, spheres, num_spheres);
              d = std::max(-1.0f, std::min(1.0f, d / TRUNC_DIST));

              int li = (lx * LEAF_DIM + ly) * LEAF_DIM + lz;
              vol.leaf_data[(size_t)off * LEAF_VOXELS + li] = d;
            }
      }

  printf("Sparse volume: %d active blocks out of %d (%.1f%% occupancy)\n",
         vol.num_active, GRID_BLOCKS,
         100.0f * vol.num_active / GRID_BLOCKS);
}

// ---------------------------------------------------------------------------
// camera
// ---------------------------------------------------------------------------

static void setup_camera(int width, int height, Camera& cam)
{
  const float target[3] = { 0.5f, 0.5f, 0.5f };
  const float yaw = 0.4f, pitch = 0.3f, dist = 2.0f;

  float f[3] = { sinf(yaw) * cosf(pitch), sinf(pitch),
                 cosf(yaw) * cosf(pitch) };
  for (int k = 0; k < 3; k++) cam.eye[k] = target[k] - dist * f[k];

  float fn = sqrtf(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
  for (int k = 0; k < 3; k++) cam.forward[k] = f[k] / fn;

  const float up_w[3] = { 0.0f, 1.0f, 0.0f };
  cam.right[0] = cam.forward[1] * up_w[2] - cam.forward[2] * up_w[1];
  cam.right[1] = cam.forward[2] * up_w[0] - cam.forward[0] * up_w[2];
  cam.right[2] = cam.forward[0] * up_w[1] - cam.forward[1] * up_w[0];
  float rn = sqrtf(cam.right[0] * cam.right[0] + cam.right[1] * cam.right[1] +
                   cam.right[2] * cam.right[2]);
  for (int k = 0; k < 3; k++) cam.right[k] /= rn;

  cam.up[0] = cam.right[1] * cam.forward[2] - cam.right[2] * cam.forward[1];
  cam.up[1] = cam.right[2] * cam.forward[0] - cam.right[0] * cam.forward[2];
  cam.up[2] = cam.right[0] * cam.forward[1] - cam.right[1] * cam.forward[0];

  cam.fov_scale = tanf(45.0f * 0.5f * 3.14159265f / 180.0f);
  cam.inv_width = 1.0f / width;
  cam.inv_height = 1.0f / height;
  cam.width = width;
  cam.height = height;
}

// ---------------------------------------------------------------------------
// ray generation and intersection helpers
// ---------------------------------------------------------------------------

static inline void generate_ray(const Camera& cam, int px, int py,
                                 float* origin, float* dir)
{
  const float aspect = (float)cam.width * cam.inv_height;
  const float u = (2.0f * (px + 0.5f) * cam.inv_width  - 1.0f) *
                  cam.fov_scale * aspect;
  const float v = (2.0f * (py + 0.5f) * cam.inv_height - 1.0f) *
                  cam.fov_scale;

  for (int k = 0; k < 3; k++) {
    origin[k] = cam.eye[k];
    dir[k] = u * cam.right[k] + v * cam.up[k] + cam.forward[k];
  }
  const float dn = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  for (int k = 0; k < 3; k++) dir[k] /= dn;
}

static inline bool ray_box(const float* origin, const float* dir,
                            float* t_near, float* t_far)
{
  float tmin = -FLT_MAX, tmax = FLT_MAX;
  for (int k = 0; k < 3; k++) {
    if (fabsf(dir[k]) < 1e-8f) {
      if (origin[k] < 0.0f || origin[k] > 1.0f) return false;
    } else {
      float inv = 1.0f / dir[k];
      float t1 = -origin[k] * inv;
      float t2 = (1.0f - origin[k]) * inv;
      if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
      tmin = std::max(tmin, t1);
      tmax = std::min(tmax, t2);
    }
  }
  *t_near = std::max(tmin, 0.0f);
  *t_far  = tmax;
  return *t_near < *t_far;
}

// ---------------------------------------------------------------------------
// TSDF lookup in the sparse volume
// ---------------------------------------------------------------------------

static inline float lookup_tsdf(const int* block_offset,
                                 const float* leaf_data,
                                 int vx, int vy, int vz)
{
  if (vx < 0 || vx >= VOXEL_DIM ||
      vy < 0 || vy >= VOXEL_DIM ||
      vz < 0 || vz >= VOXEL_DIM) return 1.0f;

  int bx = vx >> LEAF_LOG2, by = vy >> LEAF_LOG2, bz = vz >> LEAF_LOG2;
  int bi = (bx * GRID_DIM + by) * GRID_DIM + bz;
  int off = block_offset[bi];
  if (off < 0) return 1.0f;

  int lx = vx & LEAF_MASK, ly = vy & LEAF_MASK, lz = vz & LEAF_MASK;
  return leaf_data[(size_t)off * LEAF_VOXELS +
                   (lx * LEAF_DIM + ly) * LEAF_DIM + lz];
}

// ---------------------------------------------------------------------------
// two-level DDA ray caster (reference)
// ---------------------------------------------------------------------------

static bool cast_ray(const float* origin, const float* dir,
                     const int* block_offset, const float* leaf_data,
                     float* hit_t, float* hit_normal)
{
  float t_near, t_far;
  if (!ray_box(origin, dir, &t_near, &t_far)) return false;

  float inv_dir[3];
  int   step[3];
  for (int k = 0; k < 3; k++) {
    inv_dir[k] = (fabsf(dir[k]) > 1e-8f) ? (1.0f / dir[k]) : 1e8f;
    step[k]    = (dir[k] >= 0.0f) ? 1 : -1;
  }

  float entry[3];
  for (int k = 0; k < 3; k++)
    entry[k] = std::max(0.0f, std::min(origin[k] + t_near * dir[k],
                                       1.0f - 1e-6f));

  int   bi[3];
  float tMax[3], tDelta[3];
  for (int k = 0; k < 3; k++) {
    bi[k] = std::min(GRID_DIM - 1, std::max(0, (int)(entry[k] / BLOCK_W)));
    float boundary = (step[k] > 0)
        ? (bi[k] + 1) * BLOCK_W
        :  bi[k]      * BLOCK_W;
    tMax[k]  = t_near + (boundary - entry[k]) * inv_dir[k];
    tDelta[k] = fabsf(BLOCK_W * inv_dir[k]);
  }

  float t_block = t_near;

  for (int s = 0; s < GRID_DIM * 3; s++) {
    if (bi[0] < 0 || bi[0] >= GRID_DIM ||
        bi[1] < 0 || bi[1] >= GRID_DIM ||
        bi[2] < 0 || bi[2] >= GRID_DIM) break;

    float t_next = std::min({ tMax[0], tMax[1], tMax[2], t_far });

    int idx = (bi[0] * GRID_DIM + bi[1]) * GRID_DIM + bi[2];
    int off = block_offset[idx];

    if (off >= 0) {
      // fine-grained DDA inside the active block
      float ve[3];
      for (int k = 0; k < 3; k++)
        ve[k] = std::max(0.0f, std::min(origin[k] + t_block * dir[k],
                                        1.0f - 1e-6f));

      float bo[3] = { bi[0] * BLOCK_W, bi[1] * BLOCK_W, bi[2] * BLOCK_W };

      int   vi[3];
      float vtMax[3], vtDelta[3];
      for (int k = 0; k < 3; k++) {
        float local = (ve[k] - bo[k]) / VOXEL_SIZE;
        vi[k] = std::min(LEAF_DIM - 1, std::max(0, (int)local));
        float vb = (step[k] > 0) ? (vi[k] + 1) * VOXEL_SIZE + bo[k]
                                 :  vi[k]      * VOXEL_SIZE + bo[k];
        vtMax[k]   = t_block + (vb - ve[k]) * inv_dir[k];
        vtDelta[k] = fabsf(VOXEL_SIZE * inv_dir[k]);
      }

      int gv[3] = { bi[0] * LEAF_DIM + vi[0],
                     bi[1] * LEAF_DIM + vi[1],
                     bi[2] * LEAF_DIM + vi[2] };

      float prev = lookup_tsdf(block_offset, leaf_data, gv[0], gv[1], gv[2]);
      float prev_t = t_block;

      for (int vs = 0; vs < LEAF_DIM * 3; vs++) {
        float cur_t;
        if (vtMax[0] <= vtMax[1] && vtMax[0] <= vtMax[2]) {
          cur_t = vtMax[0]; vi[0] += step[0]; gv[0] += step[0];
          vtMax[0] += vtDelta[0];
        } else if (vtMax[1] <= vtMax[2]) {
          cur_t = vtMax[1]; vi[1] += step[1]; gv[1] += step[1];
          vtMax[1] += vtDelta[1];
        } else {
          cur_t = vtMax[2]; vi[2] += step[2]; gv[2] += step[2];
          vtMax[2] += vtDelta[2];
        }
        if (vi[0] < 0 || vi[0] >= LEAF_DIM ||
            vi[1] < 0 || vi[1] >= LEAF_DIM ||
            vi[2] < 0 || vi[2] >= LEAF_DIM ||
            cur_t > t_next) break;

        float cur = lookup_tsdf(block_offset, leaf_data,
                                gv[0], gv[1], gv[2]);

        if (prev > 0.0f && cur <= 0.0f) {
          float alpha = prev / (prev - cur);
          *hit_t = prev_t + alpha * (cur_t - prev_t);

          float nx = lookup_tsdf(block_offset, leaf_data, gv[0]+1, gv[1], gv[2])
                   - lookup_tsdf(block_offset, leaf_data, gv[0]-1, gv[1], gv[2]);
          float ny = lookup_tsdf(block_offset, leaf_data, gv[0], gv[1]+1, gv[2])
                   - lookup_tsdf(block_offset, leaf_data, gv[0], gv[1]-1, gv[2]);
          float nz = lookup_tsdf(block_offset, leaf_data, gv[0], gv[1], gv[2]+1)
                   - lookup_tsdf(block_offset, leaf_data, gv[0], gv[1], gv[2]-1);
          float nn = sqrtf(nx * nx + ny * ny + nz * nz);
          if (nn > 1e-6f) { nx /= nn; ny /= nn; nz /= nn; }
          else            { nx = 0; ny = 1; nz = 0; }
          hit_normal[0] = nx; hit_normal[1] = ny; hit_normal[2] = nz;
          return true;
        }
        prev   = cur;
        prev_t = cur_t;
      }
    }

    t_block = t_next;
    if      (tMax[0] <= tMax[1] && tMax[0] <= tMax[2]) { bi[0] += step[0]; tMax[0] += tDelta[0]; }
    else if (tMax[1] <= tMax[2])                       { bi[1] += step[1]; tMax[1] += tDelta[1]; }
    else                                               { bi[2] += step[2]; tMax[2] += tDelta[2]; }
  }
  return false;
}

// ---------------------------------------------------------------------------
// reference renderer
// ---------------------------------------------------------------------------

static void reference_render(const Camera& cam,
                              const int* block_offset,
                              const float* leaf_data,
                              float* image)
{
  const float light[3] = { 0.57735f, 0.57735f, 0.57735f };

  for (int py = 0; py < cam.height; py++)
    for (int px = 0; px < cam.width; px++) {
      float origin[3], dir[3];
      generate_ray(cam, px, py, origin, dir);

      float ht, hn[3];
      size_t o = 4 * ((size_t)py * cam.width + px);

      if (cast_ray(origin, dir, block_offset, leaf_data, &ht, hn)) {
        float ndotl = hn[0] * light[0] + hn[1] * light[1] + hn[2] * light[2];
        float shade = 0.2f + 0.8f * std::max(0.0f, ndotl);
        image[o + 0] = shade;
        image[o + 1] = shade;
        image[o + 2] = shade;
        image[o + 3] = ht;
      } else {
        image[o + 0] = BG_COLOR;
        image[o + 1] = BG_COLOR;
        image[o + 2] = BG_COLOR;
        image[o + 3] = 0.0f;
      }
    }
}

// ---------------------------------------------------------------------------
// verification
// ---------------------------------------------------------------------------

static bool close_enough(float a, float b, float tol)
{
  return fabsf(a - b) <= tol * (1.0f + fabsf(b));
}

#endif // REFERENCE_H
