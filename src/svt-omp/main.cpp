#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>

#include "reference.h"

#pragma omp declare target

static inline void d_generate_ray(const Camera& cam, int px, int py,
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
  float inv_dn = 1.0f / sqrtf(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
  for (int k = 0; k < 3; k++) dir[k] *= inv_dn;
}

static inline bool d_ray_box(const float* origin, const float* dir,
                              float* t_near, float* t_far)
{
  float tmin = -1e30f, tmax = 1e30f;
  for (int k = 0; k < 3; k++) {
    if (fabsf(dir[k]) < 1e-8f) {
      if (origin[k] < 0.0f || origin[k] > 1.0f) return false;
    } else {
      float inv = 1.0f / dir[k];
      float t1 = -origin[k] * inv;
      float t2 = (1.0f - origin[k]) * inv;
      if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
      if (t1 > tmin) tmin = t1;
      if (t2 < tmax) tmax = t2;
    }
  }
  *t_near = (tmin > 0.0f) ? tmin : 0.0f;
  *t_far  = tmax;
  return *t_near < *t_far;
}

static inline float d_lookup_tsdf(const int* block_offset,
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

static bool d_cast_ray(const float* origin, const float* dir,
                        const int* block_offset, const float* leaf_data,
                        float* hit_t, float* hit_normal)
{
  float t_near, t_far;
  if (!d_ray_box(origin, dir, &t_near, &t_far)) return false;

  float inv_dir[3];
  int   step[3];
  for (int k = 0; k < 3; k++) {
    inv_dir[k] = (fabsf(dir[k]) > 1e-8f) ? (1.0f / dir[k]) : 1e8f;
    step[k]    = (dir[k] >= 0.0f) ? 1 : -1;
  }

  float entry[3];
  for (int k = 0; k < 3; k++) {
    entry[k] = origin[k] + t_near * dir[k];
    if (entry[k] < 0.0f) entry[k] = 0.0f;
    if (entry[k] > 1.0f - 1e-6f) entry[k] = 1.0f - 1e-6f;
  }

  int   bi[3];
  float tMax[3], tDelta[3];
  for (int k = 0; k < 3; k++) {
    bi[k] = (int)(entry[k] / BLOCK_W);
    if (bi[k] < 0) bi[k] = 0;
    if (bi[k] >= GRID_DIM) bi[k] = GRID_DIM - 1;
    float boundary = (step[k] > 0) ? (bi[k] + 1) * BLOCK_W
                                   :  bi[k]      * BLOCK_W;
    tMax[k]   = t_near + (boundary - entry[k]) * inv_dir[k];
    tDelta[k] = fabsf(BLOCK_W * inv_dir[k]);
  }

  float t_block = t_near;

  for (int s = 0; s < GRID_DIM * 3; s++) {
    if (bi[0] < 0 || bi[0] >= GRID_DIM ||
        bi[1] < 0 || bi[1] >= GRID_DIM ||
        bi[2] < 0 || bi[2] >= GRID_DIM) break;

    float t_next = tMax[0];
    if (tMax[1] < t_next) t_next = tMax[1];
    if (tMax[2] < t_next) t_next = tMax[2];
    if (t_far   < t_next) t_next = t_far;

    int idx = (bi[0] * GRID_DIM + bi[1]) * GRID_DIM + bi[2];
    int off = block_offset[idx];

    if (off >= 0) {
      float ve[3];
      for (int k = 0; k < 3; k++) {
        ve[k] = origin[k] + t_block * dir[k];
        if (ve[k] < 0.0f) ve[k] = 0.0f;
        if (ve[k] > 1.0f - 1e-6f) ve[k] = 1.0f - 1e-6f;
      }

      float bo[3] = { bi[0] * BLOCK_W, bi[1] * BLOCK_W, bi[2] * BLOCK_W };
      int   vi[3];
      float vtMax[3], vtDelta[3];
      for (int k = 0; k < 3; k++) {
        float local = (ve[k] - bo[k]) / VOXEL_SIZE;
        vi[k] = (int)local;
        if (vi[k] < 0) vi[k] = 0;
        if (vi[k] >= LEAF_DIM) vi[k] = LEAF_DIM - 1;
        float vb = (step[k] > 0) ? (vi[k] + 1) * VOXEL_SIZE + bo[k]
                                 :  vi[k]      * VOXEL_SIZE + bo[k];
        vtMax[k]   = t_block + (vb - ve[k]) * inv_dir[k];
        vtDelta[k] = fabsf(VOXEL_SIZE * inv_dir[k]);
      }

      int gv[3] = { bi[0] * LEAF_DIM + vi[0],
                     bi[1] * LEAF_DIM + vi[1],
                     bi[2] * LEAF_DIM + vi[2] };

      const float* leaf_base = leaf_data + (size_t)off * LEAF_VOXELS;
      float prev   = leaf_base[(vi[0] * LEAF_DIM + vi[1]) * LEAF_DIM + vi[2]];
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
        const bool left_leaf =
            vi[0] < 0 || vi[0] >= LEAF_DIM ||
            vi[1] < 0 || vi[1] >= LEAF_DIM ||
            vi[2] < 0 || vi[2] >= LEAF_DIM;

        // When the fine DDA steps out of the current leaf, gv[] already
        // holds the global coordinate of the neighboring voxel. Sample it
        // through the global lookup (which transparently resolves the
        // neighboring block/leaf) so a surface crossing that straddles the
        // leaf boundary is not silently discarded.
        float cur = left_leaf
            ? d_lookup_tsdf(block_offset, leaf_data, gv[0], gv[1], gv[2])
            : leaf_base[(vi[0] * LEAF_DIM + vi[1]) * LEAF_DIM + vi[2]];

        if (prev > 0.0f && cur <= 0.0f) {
          float alpha = prev / (prev - cur);
          *hit_t = prev_t + alpha * (cur_t - prev_t);

          float nx = d_lookup_tsdf(block_offset, leaf_data, gv[0]+1, gv[1], gv[2])
                   - d_lookup_tsdf(block_offset, leaf_data, gv[0]-1, gv[1], gv[2]);
          float ny = d_lookup_tsdf(block_offset, leaf_data, gv[0], gv[1]+1, gv[2])
                   - d_lookup_tsdf(block_offset, leaf_data, gv[0], gv[1]-1, gv[2]);
          float nz = d_lookup_tsdf(block_offset, leaf_data, gv[0], gv[1], gv[2]+1)
                   - d_lookup_tsdf(block_offset, leaf_data, gv[0], gv[1], gv[2]-1);
          float nn2 = nx*nx + ny*ny + nz*nz;
          if (nn2 > 1e-12f) { float inv_nn = 1.0f / sqrtf(nn2); nx *= inv_nn; ny *= inv_nn; nz *= inv_nn; }
          else              { nx = 0; ny = 1; nz = 0; }
          hit_normal[0] = nx; hit_normal[1] = ny; hit_normal[2] = nz;
          return true;
        }
        prev   = cur;
        prev_t = cur_t;

        if (left_leaf || cur_t > t_next) break;
      }
    }

    t_block = t_next;
    if      (tMax[0] <= tMax[1] && tMax[0] <= tMax[2]) { bi[0] += step[0]; tMax[0] += tDelta[0]; }
    else if (tMax[1] <= tMax[2])                       { bi[1] += step[1]; tMax[1] += tDelta[1]; }
    else                                               { bi[2] += step[2]; tMax[2] += tDelta[2]; }
  }
  return false;
}

static inline void d_render_pixel(const Camera& cam, int px, int py,
                                  const int* block_offset,
                                  const float* leaf_data, float* image)
{
  float origin[3], dir[3];
  d_generate_ray(cam, px, py, origin, dir);

  float ht, hn[3];
  const size_t o = 4 * ((size_t)py * cam.width + px);
  if (d_cast_ray(origin, dir, block_offset, leaf_data, &ht, hn)) {
    const float light[3] = { 0.57735f, 0.57735f, 0.57735f };
    const float ndotl = hn[0]*light[0] + hn[1]*light[1] + hn[2]*light[2];
    const float shade = 0.2f + 0.8f * ((ndotl > 0.0f) ? ndotl : 0.0f);
    image[o+0] = shade; image[o+1] = shade;
    image[o+2] = shade; image[o+3] = ht;
  } else {
    image[o+0] = BG_COLOR; image[o+1] = BG_COLOR;
    image[o+2] = BG_COLOR; image[o+3] = 0.0f;
  }
}

#pragma omp end declare target

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
  int num_spheres = 12, width = 1280, height = 720, reps = 100;
  if (argc > 1) num_spheres = atoi(argv[1]);
  if (argc > 2) width       = atoi(argv[2]);
  if (argc > 3) height      = atoi(argv[3]);
  if (argc > 4) reps        = atoi(argv[4]);
  if (num_spheres > MAX_SPHERES) num_spheres = MAX_SPHERES;
  if (num_spheres <= 0 || width <= 0 || height <= 0 || reps <= 0) {
    fprintf(stderr, "Spheres, width, height, and reps must all be positive\n");
    return 1;
  }

  printf("Spheres=%d  Image=%dx%d  Reps=%d\n", num_spheres, width, height, reps);

  Sphere spheres[MAX_SPHERES];
  generate_spheres(spheres, num_spheres);

  SparseVolume vol;
  build_volume(vol, spheres, num_spheres);

  Camera cam;
  setup_camera(width, height, cam);

  const size_t npix = (size_t)width * height;

  std::vector<float> ref(4 * npix, 0.0f);
  reference_render(cam, vol.block_offset, vol.leaf_data.data(), ref.data());

  int ref_hits = 0;
  for (size_t i = 0; i < npix; i++)
    if (ref[4 * i + 3] > 0.0f) ref_hits++;
  printf("Reference: %d / %zu pixels hit a surface\n", ref_hits, npix);

  const int   *h_block = vol.block_offset;
  const float *h_leaf  = vol.leaf_data.data();

  std::vector<float> gpu(4 * npix, 0.0f);
  float *h_image = gpu.data();
  bool pass = false;

  const int num_teams = ((height + BLOCK_Y - 1) / BLOCK_Y) *
                        ((width  + BLOCK_X - 1) / BLOCK_X);
  const int thread_limit = BLOCK_X * BLOCK_Y;

  #pragma omp target data map(to: h_block[0:GRID_BLOCKS]) \
                           map(to: h_leaf[0:vol.num_active * LEAF_VOXELS]) \
                           map(tofrom: h_image[0:4 * npix])
  {
    // warmup
    for (int r = 0; r < reps; r++) {
      #pragma omp target teams distribute parallel for collapse(2) \
              num_teams(num_teams) thread_limit(thread_limit) //nowait
      for (int py = 0; py < height; py++)
        for (int px = 0; px < width; px++)
          d_render_pixel(cam, px, py, h_block, h_leaf, h_image);
    }
    //#pragma omp taskwait

    // verify
    #pragma omp target update from(h_image[0:4 * npix])

    int mismatches = 0;
    for (size_t i = 0; i < npix; i++) {
      bool pixel_mismatch = false;
      for (int c = 0; c < 4; c++)
        pixel_mismatch |= !close_enough(gpu[4 * i + c], ref[4 * i + c], 1e-2f);
      mismatches += pixel_mismatch;
    }

    const int allowed = (int)(npix * 1e-4) + 1;
    bool pass_result = (mismatches <= allowed);
    printf("Raycast: %s\n", pass_result ? "PASS" : "FAIL");
    if (mismatches > 0)
      printf("  %d / %zu pixels differ (allowed %d)\n",
             mismatches, npix, allowed);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; r++) {
      #pragma omp target teams distribute parallel for collapse(2) \
              num_teams(num_teams) thread_limit(thread_limit) //nowait
      for (int py = 0; py < height; py++)
        for (int px = 0; px < width; px++)
          d_render_pixel(cam, px, py, h_block, h_leaf, h_image);
    }
    //#pragma omp taskwait
    auto t1 = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    printf("Average Raycast time: %.3f ms\n", ms / reps);

    pass = pass_result;
  }

  return 0;
}
