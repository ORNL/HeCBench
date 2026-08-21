#ifndef REFERENCE_H
#define REFERENCE_H

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <algorithm>
#include <vector>

// 4D Gaussian splatting of a deformable scene, as used by the physics-aware
// endoscopic simulators built on 4DGS and the material point method
// (EndoGSim, https://arxiv.org/abs/2605.16022). Three stages are benchmarked:
//
//   1. an MLS-MPM step (Hu et al., https://arxiv.org/abs/1806.01923) that
//      advects the Gaussians as material particles: particle-to-grid
//      scatter, grid update, grid-to-particle gather;
//   2. the 4D preprocess, which conditions every 4D Gaussian on the current
//      time, projects the conditional 3D covariance to a 2D conic with the
//      EWA Jacobian (Zwicker et al., 2001) and evaluates the view-dependent
//      colour from spherical harmonics;
//   3. the tile-based rasterizer, which alpha-blends the Gaussians of each
//      16x16 tile front to back (Kerbl et al., https://arxiv.org/abs/2308.04079).
//
// The 4D Gaussian follows the native formulation of Yang et al.
// (https://arxiv.org/abs/2310.10642): a 4x4 covariance built from a pair of
// isoclinic quaternions, conditioned on t to give a 3D Gaussian whose opacity
// decays with the temporal distance.

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

// spherical harmonics up to degree 3 (16 coefficients per colour channel)
#define SH_DEGREE 3
#define SH_COEFFS 16

// MLS-MPM background grid
#define MPM_GRID 64
#define MPM_CELLS (MPM_GRID * MPM_GRID * MPM_GRID)

// The particles are binned into MPM_BLOCK^3 cell blocks so that the
// particle-to-grid scatter can accumulate in a block local tile instead of
// hammering the global grid with atomics. A particle in a block writes to
// cells [start - 1, start + MPM_BLOCK + 1], hence the tile is three cells
// wider than the block.
#define MPM_BLOCK 4
#define MPM_TILE (MPM_BLOCK + 3)
#define MPM_TILE_CELLS (MPM_TILE * MPM_TILE * MPM_TILE)
#define MPM_BLOCKS_PER_DIM (MPM_GRID / MPM_BLOCK)
// the largest number of particles one thread block scatters in one pass
#define MPM_CHUNK 256

// the transmittance below which a pixel stops accumulating
#define MIN_TRANSMITTANCE 1e-4f
// Gaussians fainter than this after the temporal decay are dropped
#define MIN_OPACITY (1.0f / 255.0f)

struct Camera {
  float view[9];      // world -> camera rotation, row major
  float cam_pos[3];   // camera centre in world space
  float focal_x, focal_y;
  float center_x, center_y;
  float near_plane;
  int width, height;
  int tiles_x, tiles_y;
};

struct MpmParams {
  float dt;
  float dx, inv_dx;
  float particle_volume;
  float particle_mass;
  float mu, lambda;   // Lame parameters of the neo-Hookean model
  float gravity;
  int   boundary;     // number of cells clamped at each wall
};

// ---------------------------------------------------------------------------
// deterministic input generation (identical on the host and every backend)
// ---------------------------------------------------------------------------

// xorshift32: reproducible without depending on the host's rand()
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

static inline float rng_signed(unsigned& state)
{
  return 2.0f * rng_uniform(state) - 1.0f;
}

static inline void normalize4(float* q)
{
  const float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const float inv = (n > 0.0f) ? 1.0f / n : 1.0f;
  for (int i = 0; i < 4; i++) q[i] *= inv;
}

// The scene is a slab of tissue in the unit cube, so that it is both a valid
// MPM domain and visible from the camera placed in front of it.
struct Scene {
  int num_gaussians;

  std::vector<float> mean;      // 4 * N: xyz in the unit cube, t in [0,1]
  std::vector<float> scale;     // 4 * N: spatial extents and the temporal one
  std::vector<float> quat_l;    // 4 * N
  std::vector<float> quat_r;    // 4 * N
  std::vector<float> opacity;   // N
  std::vector<float> sh;        // SH_COEFFS * 3 * N, coefficient major

  std::vector<float> velocity;  // 3 * N
  std::vector<float> affine;    // 9 * N: the APIC velocity gradient C
  std::vector<float> defgrad;   // 9 * N: the deformation gradient F

  // particle chunks, each holding up to MPM_CHUNK particles of one cell block
  std::vector<int> chunk_start;  // num_chunks + 1
  std::vector<int> chunk_block;  // num_chunks, the linear cell block index
};

static void reorder_particles(Scene& s, const std::vector<int>& order)
{
  const int n = s.num_gaussians;
  const Scene t = s;
  for (int i = 0; i < n; i++) {
    const int o = order[i];
    for (int k = 0; k < 4; k++) {
      s.mean[4 * (size_t)i + k] = t.mean[4 * (size_t)o + k];
      s.scale[4 * (size_t)i + k] = t.scale[4 * (size_t)o + k];
      s.quat_l[4 * (size_t)i + k] = t.quat_l[4 * (size_t)o + k];
      s.quat_r[4 * (size_t)i + k] = t.quat_r[4 * (size_t)o + k];
    }
    s.opacity[i] = t.opacity[o];
    for (int k = 0; k < 3; k++)
      s.velocity[3 * (size_t)i + k] = t.velocity[3 * (size_t)o + k];
    for (int k = 0; k < 9; k++) {
      s.affine[9 * (size_t)i + k] = t.affine[9 * (size_t)o + k];
      s.defgrad[9 * (size_t)i + k] = t.defgrad[9 * (size_t)o + k];
    }
    for (int c = 0; c < SH_COEFFS * 3; c++)
      s.sh[(size_t)c * n + i] = t.sh[(size_t)c * n + o];
  }
}

static void generate_scene(int n, Scene& s)
{
  s.num_gaussians = n;
  s.mean.resize((size_t)4 * n);
  s.scale.resize((size_t)4 * n);
  s.quat_l.resize((size_t)4 * n);
  s.quat_r.resize((size_t)4 * n);
  s.opacity.resize(n);
  s.sh.resize((size_t)SH_COEFFS * 3 * n);
  s.velocity.resize((size_t)3 * n);
  s.affine.resize((size_t)9 * n);
  s.defgrad.resize((size_t)9 * n);

  unsigned state = 123456789u;

  for (int i = 0; i < n; i++) {
    // particles fill the middle of the grid, away from the walls
    s.mean[4 * (size_t)i + 0] = 0.2f + 0.6f * rng_uniform(state);
    s.mean[4 * (size_t)i + 1] = 0.2f + 0.4f * rng_uniform(state);
    s.mean[4 * (size_t)i + 2] = 0.2f + 0.6f * rng_uniform(state);
    s.mean[4 * (size_t)i + 3] = rng_uniform(state);          // time of birth

    // small anisotropic splats, and a temporal extent that keeps a useful
    // fraction of the scene alive at any one time
    s.scale[4 * (size_t)i + 0] = 0.004f + 0.010f * rng_uniform(state);
    s.scale[4 * (size_t)i + 1] = 0.004f + 0.010f * rng_uniform(state);
    s.scale[4 * (size_t)i + 2] = 0.004f + 0.010f * rng_uniform(state);
    s.scale[4 * (size_t)i + 3] = 0.03f + 0.15f * rng_uniform(state);

    float ql[4], qr[4];
    for (int k = 0; k < 4; k++) ql[k] = rng_signed(state);
    normalize4(ql);
    // qr near conj(ql) keeps the rotation close to a spatial one; the
    // perturbation is the space-time coupling that moves the Gaussian
    qr[0] = ql[0] + 0.25f * rng_signed(state);
    for (int k = 1; k < 4; k++) qr[k] = -ql[k] + 0.25f * rng_signed(state);
    normalize4(qr);
    for (int k = 0; k < 4; k++) s.quat_l[4 * (size_t)i + k] = ql[k];
    for (int k = 0; k < 4; k++) s.quat_r[4 * (size_t)i + k] = qr[k];

    s.opacity[i] = 0.15f + 0.75f * rng_uniform(state);

    // the band 0 coefficient carries the base colour, the higher bands the
    // view dependent sheen of wet tissue
    for (int c = 0; c < SH_COEFFS; c++) {
      const float amp = (c == 0) ? 0.5f : 0.15f / (float)(1 + c);
      for (int ch = 0; ch < 3; ch++)
        s.sh[((size_t)c * 3 + ch) * n + i] = amp * rng_signed(state);
    }

    // a slow swirl, so that the first step already deforms the material
    const float rx = s.mean[4 * (size_t)i + 0] - 0.5f;
    const float rz = s.mean[4 * (size_t)i + 2] - 0.5f;
    s.velocity[3 * (size_t)i + 0] = -2.0f * rz;
    s.velocity[3 * (size_t)i + 1] = 0.0f;
    s.velocity[3 * (size_t)i + 2] = 2.0f * rx;

    for (int k = 0; k < 9; k++) s.affine[9 * (size_t)i + k] = 0.0f;
    for (int k = 0; k < 9; k++)
      s.defgrad[9 * (size_t)i + k] = (k % 4 == 0) ? 1.0f : 0.0f;
  }

  // Bin the particles by MPM_BLOCK^3 cell block, then by cell. Production MPM
  // codes keep the particles binned for exactly this reason: it bounds the
  // grid footprint of a thread block, which is what lets the scatter stay in
  // local memory, and it keeps the gather reading a few cache lines.
  std::vector<int> order(n);
  std::vector<int> key(n);
  std::vector<int> block(n);
  for (int i = 0; i < n; i++) {
    order[i] = i;
    const int ix = std::min(MPM_GRID - 1, std::max(0, (int)(s.mean[4 * (size_t)i + 0] * MPM_GRID)));
    const int iy = std::min(MPM_GRID - 1, std::max(0, (int)(s.mean[4 * (size_t)i + 1] * MPM_GRID)));
    const int iz = std::min(MPM_GRID - 1, std::max(0, (int)(s.mean[4 * (size_t)i + 2] * MPM_GRID)));
    block[i] = ((ix / MPM_BLOCK) * MPM_BLOCKS_PER_DIM + (iy / MPM_BLOCK)) *
                   MPM_BLOCKS_PER_DIM + (iz / MPM_BLOCK);
    key[i] = (ix * MPM_GRID + iy) * MPM_GRID + iz;
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return (block[a] != block[b]) ? (block[a] < block[b]) : (key[a] < key[b]);
  });

  reorder_particles(s, order);

  // cut the sorted particles into chunks that hold one cell block each
  s.chunk_start.clear();
  s.chunk_block.clear();
  int begin = 0;
  while (begin < n) {
    const int b = block[order[begin]];
    int end = begin;
    while (end < n && block[order[end]] == b && end - begin < MPM_CHUNK) end++;
    s.chunk_start.push_back(begin);
    s.chunk_block.push_back(b);
    begin = end;
  }
  s.chunk_start.push_back(n);

  // Within a chunk the particles are still sorted by cell, so neighbouring
  // lanes would scatter into the same cell and serialize on it. Interleaving
  // the chunk hands each lane a particle from a different cell while keeping
  // the loads of a warp contiguous.
  std::vector<int> shuffled(n);
  const int lanes = 32;
  for (size_t c = 0; c + 1 < s.chunk_start.size(); c++) {
    const int cb = s.chunk_start[c], ce = s.chunk_start[c + 1];
    const int len = ce - cb;
    const int rows = (len + lanes - 1) / lanes;
    int out = cb;
    for (int r = 0; r < rows; r++)
      for (int l = 0; l < lanes; l++) {
        const int src = l * rows + r;
        if (src < len) shuffled[out++] = cb + src;
      }
  }
  reorder_particles(s, shuffled);
}

static void setup_camera(int width, int height, Camera& cam)
{
  // Look at the centre of the tissue block from a yawed and slightly raised
  // position, so that the view rotation exercises all nine entries.
  const float target[3] = { 0.5f, 0.4f, 0.5f };
  const float yaw = 0.35f, pitch = 0.20f, dist = 1.6f;

  // f points from the camera to the target
  const float f[3] = { sinf(yaw) * cosf(pitch), -sinf(pitch),
                       cosf(yaw) * cosf(pitch) };
  for (int k = 0; k < 3; k++) cam.cam_pos[k] = target[k] - dist * f[k];

  // r = up x f, u = f x r
  const float up[3] = { 0.0f, 1.0f, 0.0f };
  float r[3] = { up[1] * f[2] - up[2] * f[1],
                 up[2] * f[0] - up[0] * f[2],
                 up[0] * f[1] - up[1] * f[0] };
  const float rn = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  for (int k = 0; k < 3; k++) r[k] /= rn;
  const float u[3] = { f[1] * r[2] - f[2] * r[1],
                       f[2] * r[0] - f[0] * r[2],
                       f[0] * r[1] - f[1] * r[0] };

  for (int k = 0; k < 3; k++) {
    cam.view[0 * 3 + k] = r[k];
    cam.view[1 * 3 + k] = u[k];
    cam.view[2 * 3 + k] = f[k];
  }

  cam.focal_x = 0.8f * width;
  cam.focal_y = 0.8f * width;
  cam.center_x = 0.5f * width;
  cam.center_y = 0.5f * height;
  cam.near_plane = 0.2f;
  cam.width = width;
  cam.height = height;
  cam.tiles_x = (width + BLOCK_X - 1) / BLOCK_X;
  cam.tiles_y = (height + BLOCK_Y - 1) / BLOCK_Y;
}

static void setup_mpm(MpmParams& p)
{
  p.dx = 1.0f / MPM_GRID;
  p.inv_dx = (float)MPM_GRID;
  p.dt = 5e-5f;
  p.particle_volume = p.dx * p.dx * p.dx * 0.25f;
  p.particle_mass = p.particle_volume * 1000.0f;
  // soft tissue: a low Young's modulus and a nearly incompressible Poisson
  const float E = 5.0e3f, nu = 0.4f;
  p.mu = E / (2.0f * (1.0f + nu));
  p.lambda = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
  p.gravity = -9.8f;
  p.boundary = 3;
  return;
}

// ---------------------------------------------------------------------------
// shared math, written so the device kernels can use exactly the same formulas
// ---------------------------------------------------------------------------

// Build the 4x4 covariance of a 4D Gaussian from the scale and the pair of
// isoclinic quaternions, and condition it on the time t. Returns the 6 unique
// entries of the conditional 3D covariance, the conditional mean and the
// opacity attenuation. Yang et al., https://arxiv.org/abs/2310.10642
static inline void condition_4d_gaussian(
    const float* mean4, const float* scale4,
    const float* ql, const float* qr, float time,
    float* mean3, float* cov3, float* opacity_scale)
{
  // Left and right isoclinic rotations. The quaternion basis is ordered
  // (t, x, y, z): with qr = conj(ql) the rotation is purely spatial, and the
  // deviation of qr from conj(ql) is what couples space and time, that is,
  // what makes the Gaussian travel as t advances.
  const float lw = ql[0], lx = ql[1], ly = ql[2], lz = ql[3];
  const float rw = qr[0], rx = qr[1], ry = qr[2], rz = qr[3];

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

  // M = L * R is the 4D rotation
  float M[16];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      float acc = 0.0f;
      for (int k = 0; k < 4; k++) acc += L[i * 4 + k] * R[k * 4 + j];
      M[i * 4 + j] = acc;
    }

  // Sigma = M diag(s^2) M^T, in the (t, x, y, z) basis
  const float s2[4] = { scale4[3] * scale4[3], scale4[0] * scale4[0],
                        scale4[1] * scale4[1], scale4[2] * scale4[2] };
  float sigma[16];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      float acc = 0.0f;
      for (int k = 0; k < 4; k++) acc += M[i * 4 + k] * s2[k] * M[j * 4 + k];
      sigma[i * 4 + j] = acc;
    }

  // condition the spatial block on the temporal one
  const float sigma_tt = sigma[0];
  const float inv_tt = 1.0f / sigma_tt;
  const float st[3] = { sigma[4], sigma[8], sigma[12] };   // Sigma_{xyz,t}
  const float dt = time - mean4[3];

  for (int i = 0; i < 3; i++) mean3[i] = mean4[i] + dt * st[i] * inv_tt;

  cov3[0] = sigma[5] - st[0] * st[0] * inv_tt;    // xx
  cov3[1] = sigma[6] - st[0] * st[1] * inv_tt;    // xy
  cov3[2] = sigma[7] - st[0] * st[2] * inv_tt;    // xz
  cov3[3] = sigma[10] - st[1] * st[1] * inv_tt;   // yy
  cov3[4] = sigma[11] - st[1] * st[2] * inv_tt;   // yz
  cov3[5] = sigma[15] - st[2] * st[2] * inv_tt;   // zz

  *opacity_scale = expf(-0.5f * dt * dt * inv_tt);
}

// Project the 3D covariance to the 2D conic with the EWA Jacobian.
// Returns false if the Gaussian is behind the near plane or degenerate.
static inline bool project_gaussian(
    const float* mean3, const float* cov3, const Camera& cam,
    float* mean2d, float* conic, float* radius, float* depth)
{
  // world -> camera
  const float p[3] = { mean3[0] - cam.cam_pos[0],
                       mean3[1] - cam.cam_pos[1],
                       mean3[2] - cam.cam_pos[2] };
  float t[3];
  for (int i = 0; i < 3; i++)
    t[i] = cam.view[i * 3 + 0] * p[0] + cam.view[i * 3 + 1] * p[1] +
           cam.view[i * 3 + 2] * p[2];

  if (t[2] < cam.near_plane) return false;

  const float inv_z = 1.0f / t[2];
  const float inv_z2 = inv_z * inv_z;

  // J is the Jacobian of the perspective projection, clamped as in EWA
  // splatting so that Gaussians far off axis stay well conditioned
  const float lim_x = 1.3f * (0.5f * cam.width) / cam.focal_x;
  const float lim_y = 1.3f * (0.5f * cam.height) / cam.focal_y;
  const float tx = t[2] * std::min(lim_x, std::max(-lim_x, t[0] * inv_z));
  const float ty = t[2] * std::min(lim_y, std::max(-lim_y, t[1] * inv_z));

  const float J[6] = { cam.focal_x * inv_z, 0.0f, -cam.focal_x * tx * inv_z2,
                       0.0f, cam.focal_y * inv_z, -cam.focal_y * ty * inv_z2 };

  // T = J * W  (2x3)
  float T[6];
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
      T[i * 3 + j] = J[i * 3 + 0] * cam.view[0 * 3 + j] +
                     J[i * 3 + 1] * cam.view[1 * 3 + j] +
                     J[i * 3 + 2] * cam.view[2 * 3 + j];

  const float S[9] = { cov3[0], cov3[1], cov3[2],
                       cov3[1], cov3[3], cov3[4],
                       cov3[2], cov3[4], cov3[5] };

  // cov2d = T S T^T, with the low pass filter of one third of a pixel
  float TS[6];
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
      TS[i * 3 + j] = T[i * 3 + 0] * S[0 * 3 + j] + T[i * 3 + 1] * S[1 * 3 + j] +
                      T[i * 3 + 2] * S[2 * 3 + j];

  float a = 0.0f, b = 0.0f, c = 0.0f;
  for (int k = 0; k < 3; k++) {
    a += TS[0 * 3 + k] * T[0 * 3 + k];
    b += TS[0 * 3 + k] * T[1 * 3 + k];
    c += TS[1 * 3 + k] * T[1 * 3 + k];
  }
  a += 0.3f;
  c += 0.3f;

  const float det = a * c - b * b;
  if (det <= 0.0f) return false;
  const float inv_det = 1.0f / det;

  conic[0] = c * inv_det;
  conic[1] = -b * inv_det;
  conic[2] = a * inv_det;

  // three sigma of the larger principal axis
  const float mid = 0.5f * (a + c);
  const float disc = sqrtf(std::max(0.1f, mid * mid - det));
  *radius = ceilf(3.0f * sqrtf(std::max(mid + disc, mid - disc)));

  mean2d[0] = cam.focal_x * t[0] * inv_z + cam.center_x;
  mean2d[1] = cam.focal_y * t[1] * inv_z + cam.center_y;
  *depth = t[2];
  return true;
}

// Real spherical harmonics up to degree 3, the basis used by every Gaussian
// splatting implementation.
// spelled out as macros so that the device kernels and the host reference
// share one definition
#define SH_C0    0.28209479177387814f
#define SH_C1    0.4886025119029199f
#define SH_C2_0  1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2  0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4  0.5462742152960396f
#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1  2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3  0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5  1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

// `sh` is coefficient major: sh[(c * 3 + channel) * n + index]
static inline void eval_sh(const float* sh, int n, int index,
                           const float* dir, float* rgb)
{
  const float x = dir[0], y = dir[1], z = dir[2];
  float basis[SH_COEFFS];

  basis[0] = SH_C0;
  basis[1] = -SH_C1 * y;
  basis[2] = SH_C1 * z;
  basis[3] = -SH_C1 * x;

  const float xx = x * x, yy = y * y, zz = z * z;
  const float xy = x * y, yz = y * z, xz = x * z;
  basis[4] = SH_C2_0 * xy;
  basis[5] = SH_C2_1 * yz;
  basis[6] = SH_C2_2 * (2.0f * zz - xx - yy);
  basis[7] = SH_C2_3 * xz;
  basis[8] = SH_C2_4 * (xx - yy);

  basis[9]  = SH_C3_0 * y * (3.0f * xx - yy);
  basis[10] = SH_C3_1 * xy * z;
  basis[11] = SH_C3_2 * y * (4.0f * zz - xx - yy);
  basis[12] = SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
  basis[13] = SH_C3_4 * x * (4.0f * zz - xx - yy);
  basis[14] = SH_C3_5 * z * (xx - yy);
  basis[15] = SH_C3_6 * x * (xx - 3.0f * yy);

  for (int ch = 0; ch < 3; ch++) {
    float acc = 0.0f;
    for (int c = 0; c < SH_COEFFS; c++)
      acc += basis[c] * sh[((size_t)c * 3 + ch) * n + index];
    rgb[ch] = std::max(acc + 0.5f, 0.0f);
  }
}

// ---------------------------------------------------------------------------
// host reference
// ---------------------------------------------------------------------------

// Stage 2: condition on time, project, shade.
static void reference_preprocess(
    const Scene& s, const Camera& cam, float time,
    float* mean2d, float* conic_opacity, float* color_depth, int* radii)
{
  const int n = s.num_gaussians;

  for (int i = 0; i < n; i++) {
    radii[i] = 0;
    mean2d[2 * (size_t)i + 0] = 0.0f;
    mean2d[2 * (size_t)i + 1] = 0.0f;
    for (int k = 0; k < 4; k++) {
      conic_opacity[4 * (size_t)i + k] = 0.0f;
      color_depth[4 * (size_t)i + k] = 0.0f;
    }

    float mean3[3], cov3[6], opacity_scale;
    condition_4d_gaussian(&s.mean[4 * (size_t)i], &s.scale[4 * (size_t)i],
                          &s.quat_l[4 * (size_t)i], &s.quat_r[4 * (size_t)i],
                          time, mean3, cov3, &opacity_scale);

    const float alpha = s.opacity[i] * opacity_scale;
    if (alpha < MIN_OPACITY) continue;

    float m2d[2], conic[3], radius, depth;
    if (!project_gaussian(mean3, cov3, cam, m2d, conic, &radius, &depth))
      continue;

    if (m2d[0] + radius < 0.0f || m2d[0] - radius > (float)cam.width ||
        m2d[1] + radius < 0.0f || m2d[1] - radius > (float)cam.height)
      continue;

    float dir[3] = { mean3[0] - cam.cam_pos[0],
                     mean3[1] - cam.cam_pos[1],
                     mean3[2] - cam.cam_pos[2] };
    const float inv_len =
        1.0f / sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (int k = 0; k < 3; k++) dir[k] *= inv_len;

    float rgb[3];
    eval_sh(s.sh.data(), n, i, dir, rgb);

    radii[i] = (int)radius;
    mean2d[2 * (size_t)i + 0] = m2d[0];
    mean2d[2 * (size_t)i + 1] = m2d[1];
    conic_opacity[4 * (size_t)i + 0] = conic[0];
    conic_opacity[4 * (size_t)i + 1] = conic[1];
    conic_opacity[4 * (size_t)i + 2] = conic[2];
    conic_opacity[4 * (size_t)i + 3] = alpha;
    color_depth[4 * (size_t)i + 0] = rgb[0];
    color_depth[4 * (size_t)i + 1] = rgb[1];
    color_depth[4 * (size_t)i + 2] = rgb[2];
    color_depth[4 * (size_t)i + 3] = depth;
  }
}

// Build the per tile lists, sorted front to back. Production implementations
// do this with a radix sort over (tile, depth) keys; here it is host side
// setup shared by every backend, so that the rasterizer sees identical input.
static void build_tile_lists(
    const Camera& cam, int n, const float* mean2d, const float* color_depth,
    const int* radii, std::vector<int>& tile_offsets, std::vector<int>& tile_list)
{
  const int num_tiles = cam.tiles_x * cam.tiles_y;
  tile_offsets.assign(num_tiles + 1, 0);

  std::vector<int> counts(num_tiles, 0);
  for (int i = 0; i < n; i++) {
    if (radii[i] <= 0) continue;
    const float r = (float)radii[i];
    const int x0 = std::max(0, (int)floorf((mean2d[2 * (size_t)i + 0] - r) / BLOCK_X));
    const int x1 = std::min(cam.tiles_x - 1, (int)floorf((mean2d[2 * (size_t)i + 0] + r) / BLOCK_X));
    const int y0 = std::max(0, (int)floorf((mean2d[2 * (size_t)i + 1] - r) / BLOCK_Y));
    const int y1 = std::min(cam.tiles_y - 1, (int)floorf((mean2d[2 * (size_t)i + 1] + r) / BLOCK_Y));
    for (int ty = y0; ty <= y1; ty++)
      for (int tx = x0; tx <= x1; tx++)
        counts[ty * cam.tiles_x + tx]++;
  }

  for (int t = 0; t < num_tiles; t++) tile_offsets[t + 1] = tile_offsets[t] + counts[t];
  tile_list.assign(tile_offsets[num_tiles], 0);

  std::vector<int> cursor(tile_offsets.begin(), tile_offsets.end() - 1);
  for (int i = 0; i < n; i++) {
    if (radii[i] <= 0) continue;
    const float r = (float)radii[i];
    const int x0 = std::max(0, (int)floorf((mean2d[2 * (size_t)i + 0] - r) / BLOCK_X));
    const int x1 = std::min(cam.tiles_x - 1, (int)floorf((mean2d[2 * (size_t)i + 0] + r) / BLOCK_X));
    const int y0 = std::max(0, (int)floorf((mean2d[2 * (size_t)i + 1] - r) / BLOCK_Y));
    const int y1 = std::min(cam.tiles_y - 1, (int)floorf((mean2d[2 * (size_t)i + 1] + r) / BLOCK_Y));
    for (int ty = y0; ty <= y1; ty++)
      for (int tx = x0; tx <= x1; tx++)
        tile_list[cursor[ty * cam.tiles_x + tx]++] = i;
  }

  for (int t = 0; t < num_tiles; t++) {
    std::sort(tile_list.begin() + tile_offsets[t],
              tile_list.begin() + tile_offsets[t + 1],
              [&](int a, int b) {
                const float da = color_depth[4 * (size_t)a + 3];
                const float db = color_depth[4 * (size_t)b + 3];
                return (da != db) ? (da < db) : (a < b);
              });
  }
}

// Stage 3: front to back alpha blending of one tile list per tile.
static void reference_render(
    const Camera& cam, const float* mean2d, const float* conic_opacity,
    const float* color_depth, const int* tile_offsets, const int* tile_list,
    float* image)
{
  for (int ty = 0; ty < cam.tiles_y; ty++) {
    for (int tx = 0; tx < cam.tiles_x; tx++) {
      const int tile = ty * cam.tiles_x + tx;
      const int begin = tile_offsets[tile], end = tile_offsets[tile + 1];

      for (int py = 0; py < BLOCK_Y; py++) {
        const int y = ty * BLOCK_Y + py;
        if (y >= cam.height) continue;
        for (int px = 0; px < BLOCK_X; px++) {
          const int x = tx * BLOCK_X + px;
          if (x >= cam.width) continue;

          float transmittance = 1.0f;
          float rgb[3] = { 0.0f, 0.0f, 0.0f };

          for (int k = begin; k < end; k++) {
            const int g = tile_list[k];
            const float dx = mean2d[2 * (size_t)g + 0] - (float)x - 0.5f;
            const float dy = mean2d[2 * (size_t)g + 1] - (float)y - 0.5f;
            const float* co = &conic_opacity[4 * (size_t)g];
            const float power =
                -0.5f * (co[0] * dx * dx + co[2] * dy * dy) - co[1] * dx * dy;
            if (power > 0.0f) continue;

            const float alpha = std::min(0.99f, co[3] * expf(power));
            if (alpha < MIN_OPACITY) continue;

            const float weight = alpha * transmittance;
            for (int ch = 0; ch < 3; ch++)
              rgb[ch] += color_depth[4 * (size_t)g + ch] * weight;

            transmittance *= 1.0f - alpha;
            if (transmittance < MIN_TRANSMITTANCE) break;
          }

          const size_t o = 4 * ((size_t)y * cam.width + x);
          image[o + 0] = rgb[0];
          image[o + 1] = rgb[1];
          image[o + 2] = rgb[2];
          image[o + 3] = 1.0f - transmittance;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// MLS-MPM reference (stage 1)
// ---------------------------------------------------------------------------

static inline void quadratic_weights(float fx, float w[3])
{
  w[0] = 0.5f * (1.5f - fx) * (1.5f - fx);
  w[1] = 0.75f - (fx - 1.0f) * (fx - 1.0f);
  w[2] = 0.5f * (fx - 0.5f) * (fx - 0.5f);
}

// The neo-Hookean first Piola-Kirchhoff stress, mapped to the MLS-MPM
// affine momentum contribution: -dt * volume * 4 * inv_dx^2 * P F^T
static inline void neo_hookean_stress(const float* F, const MpmParams& p,
                                      float* stress)
{
  const float J =
      F[0] * (F[4] * F[8] - F[5] * F[7]) -
      F[1] * (F[3] * F[8] - F[5] * F[6]) +
      F[2] * (F[3] * F[7] - F[4] * F[6]);
  const float safe_J = (fabsf(J) < 1e-6f) ? ((J < 0.0f) ? -1e-6f : 1e-6f) : J;
  const float inv_J = 1.0f / safe_J;

  // cofactor matrix = J * F^{-T}
  const float C[9] = {
     (F[4] * F[8] - F[5] * F[7]), -(F[3] * F[8] - F[5] * F[6]),  (F[3] * F[7] - F[4] * F[6]),
    -(F[1] * F[8] - F[2] * F[7]),  (F[0] * F[8] - F[2] * F[6]), -(F[0] * F[7] - F[1] * F[6]),
     (F[1] * F[5] - F[2] * F[4]), -(F[0] * F[5] - F[2] * F[3]),  (F[0] * F[4] - F[1] * F[3]) };

  // P = mu (F - F^{-T}) + lambda log(J) F^{-T}
  const float coeff = (p.lambda * logf(fabsf(safe_J)) - p.mu) * inv_J;
  float P[9];
  for (int k = 0; k < 9; k++) P[k] = p.mu * F[k] + coeff * C[k];

  // stress = -dt * volume * 4 * inv_dx^2 * P F^T
  const float s = -p.dt * p.particle_volume * 4.0f * p.inv_dx * p.inv_dx;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      float acc = 0.0f;
      for (int k = 0; k < 3; k++) acc += P[i * 3 + k] * F[j * 3 + k];
      stress[i * 3 + j] = s * acc;
    }
}

static void reference_p2g(const Scene& s, const MpmParams& p, float* grid)
{
  const int n = s.num_gaussians;
  memset(grid, 0, sizeof(float) * 4 * (size_t)MPM_CELLS);

  for (int i = 0; i < n; i++) {
    const float* x = &s.mean[4 * (size_t)i];
    const float* v = &s.velocity[3 * (size_t)i];
    const float* C = &s.affine[9 * (size_t)i];
    const float* F0 = &s.defgrad[9 * (size_t)i];

    // F <- (I + dt C) F
    float F[9];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        float acc = F0[a * 3 + b];
        for (int k = 0; k < 3; k++)
          acc += p.dt * C[a * 3 + k] * F0[k * 3 + b];
        F[a * 3 + b] = acc;
      }

    float stress[9];
    neo_hookean_stress(F, p, stress);

    // affine = stress + mass * C
    float affine[9];
    for (int k = 0; k < 9; k++) affine[k] = stress[k] + p.particle_mass * C[k];

    const float gx = x[0] * p.inv_dx, gy = x[1] * p.inv_dx, gz = x[2] * p.inv_dx;
    const int bx = (int)floorf(gx - 0.5f);
    const int by = (int)floorf(gy - 0.5f);
    const int bz = (int)floorf(gz - 0.5f);
    float wx[3], wy[3], wz[3];
    quadratic_weights(gx - bx, wx);
    quadratic_weights(gy - by, wy);
    quadratic_weights(gz - bz, wz);

    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++)
        for (int c = 0; c < 3; c++) {
          const int ix = bx + a, iy = by + b, iz = bz + c;
          if (ix < 0 || ix >= MPM_GRID || iy < 0 || iy >= MPM_GRID ||
              iz < 0 || iz >= MPM_GRID)
            continue;
          const float w = wx[a] * wy[b] * wz[c];
          const float dpos[3] = { ((float)a - (gx - bx)) * p.dx,
                                  ((float)b - (gy - by)) * p.dx,
                                  ((float)c - (gz - bz)) * p.dx };
          const size_t o = 4 * ((size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz);
          for (int k = 0; k < 3; k++) {
            const float impulse = affine[k * 3 + 0] * dpos[0] +
                                  affine[k * 3 + 1] * dpos[1] +
                                  affine[k * 3 + 2] * dpos[2];
            grid[o + k] += w * (p.particle_mass * v[k] + impulse);
          }
          grid[o + 3] += w * p.particle_mass;
        }
  }
}

static void reference_grid_update(const MpmParams& p, float* grid)
{
  for (int ix = 0; ix < MPM_GRID; ix++)
    for (int iy = 0; iy < MPM_GRID; iy++)
      for (int iz = 0; iz < MPM_GRID; iz++) {
        const size_t o = 4 * ((size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz);
        const float mass = grid[o + 3];
        if (mass <= 0.0f) {
          grid[o + 0] = grid[o + 1] = grid[o + 2] = 0.0f;
          continue;
        }
        const float inv_mass = 1.0f / mass;
        float v[3] = { grid[o + 0] * inv_mass, grid[o + 1] * inv_mass,
                       grid[o + 2] * inv_mass };
        v[1] += p.dt * p.gravity;

        // sticky walls
        if (ix < p.boundary && v[0] < 0.0f) v[0] = 0.0f;
        if (ix >= MPM_GRID - p.boundary && v[0] > 0.0f) v[0] = 0.0f;
        if (iy < p.boundary && v[1] < 0.0f) v[1] = 0.0f;
        if (iy >= MPM_GRID - p.boundary && v[1] > 0.0f) v[1] = 0.0f;
        if (iz < p.boundary && v[2] < 0.0f) v[2] = 0.0f;
        if (iz >= MPM_GRID - p.boundary && v[2] > 0.0f) v[2] = 0.0f;

        for (int k = 0; k < 3; k++) grid[o + k] = v[k];
      }
}

static void reference_g2p(Scene& s, const MpmParams& p, const float* grid)
{
  const int n = s.num_gaussians;

  for (int i = 0; i < n; i++) {
    float* x = &s.mean[4 * (size_t)i];
    float* v = &s.velocity[3 * (size_t)i];
    float* C = &s.affine[9 * (size_t)i];
    float* F0 = &s.defgrad[9 * (size_t)i];

    const float gx = x[0] * p.inv_dx, gy = x[1] * p.inv_dx, gz = x[2] * p.inv_dx;
    const int bx = (int)floorf(gx - 0.5f);
    const int by = (int)floorf(gy - 0.5f);
    const int bz = (int)floorf(gz - 0.5f);
    float wx[3], wy[3], wz[3];
    quadratic_weights(gx - bx, wx);
    quadratic_weights(gy - by, wy);
    quadratic_weights(gz - bz, wz);

    float new_v[3] = { 0.0f, 0.0f, 0.0f };
    float new_C[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++)
        for (int c = 0; c < 3; c++) {
          const int ix = bx + a, iy = by + b, iz = bz + c;
          if (ix < 0 || ix >= MPM_GRID || iy < 0 || iy >= MPM_GRID ||
              iz < 0 || iz >= MPM_GRID)
            continue;
          const float w = wx[a] * wy[b] * wz[c];
          const size_t o = 4 * ((size_t)(ix * MPM_GRID + iy) * MPM_GRID + iz);
          const float gv[3] = { grid[o + 0], grid[o + 1], grid[o + 2] };
          const float dpos[3] = { ((float)a - (gx - bx)) * p.dx,
                                  ((float)b - (gy - by)) * p.dx,
                                  ((float)c - (gz - bz)) * p.dx };
          for (int k = 0; k < 3; k++) {
            new_v[k] += w * gv[k];
            for (int l = 0; l < 3; l++)
              new_C[k * 3 + l] += 4.0f * p.inv_dx * p.inv_dx * w * gv[k] * dpos[l];
          }
        }

    // F <- (I + dt C) F, with the updated C
    float F[9];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        float acc = F0[a * 3 + b];
        for (int k = 0; k < 3; k++)
          acc += p.dt * new_C[a * 3 + k] * F0[k * 3 + b];
        F[a * 3 + b] = acc;
      }

    for (int k = 0; k < 3; k++) {
      v[k] = new_v[k];
      x[k] += p.dt * new_v[k];
    }
    for (int k = 0; k < 9; k++) {
      C[k] = new_C[k];
      F0[k] = F[k];
    }
  }
}

static bool close_enough(float a, float b, float tol)
{
  return fabsf(a - b) <= tol * (1.0f + fabsf(b));
}

#endif
