#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include "constants.h"
#include "grid.h"

#define R 4
#define NDIM 8

__attribute__ ((always_inline))
void target_inner_3d_kernel(
    llint nx, llint ny, llint nz, int ldimx, int ldimy, int ldimz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    float coef0,
    float coefx_1, float coefx_2, float coefx_3, float coefx_4,
    float coefy_1, float coefy_2, float coefy_3, float coefy_4,
    float coefz_1, float coefz_2, float coefz_3, float coefz_4,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    const float *__restrict__ phi, const float *__restrict__ eta,
    const sycl::nd_item<3> &item, sycl::local_accessor<float, 3> s_u)
{

  const llint i0 = x3 + item.get_group(0) * item.get_local_range(0);
  const llint j0 = y3 + item.get_group(1) * item.get_local_range(1);
  const llint k0 = z3 + item.get_group(2) * item.get_local_range(2);

  const int ti = item.get_local_id(0);
  const int tj = item.get_local_id(1);
  const int tk = item.get_local_id(2);

  const llint i = i0 + ti;
  const llint j = j0 + tj;
  const llint k = k0 + tk;

  s_u[ti][tj][tk] = 0.f;

  if (ti < 2*R && tj < 2*R && tk< 2*R)
    s_u[NDIM+ti][NDIM+tj][NDIM+tk] = 0.f;

  item.barrier(sycl::access::fence_space::local_space);

  const llint sui = ti + R;
  const llint suj = tj + R;
  const llint suk = tk + R;

  const int z_side = ti / R;
  s_u[ti+z_side*NDIM][suj][suk] = u[IDX3(i+(z_side*2-1)*R,j,k)];
  const int y_side = tj / R;
  s_u[sui][tj+y_side*NDIM][suk] = u[IDX3(i,j+(y_side*2-1)*R,k)];
  s_u[sui][suj][tk] = u[IDX3(i,j,k-R)];
  s_u[sui][suj][tk+NDIM] = u[IDX3(i,j,k+R)];

  item.barrier(sycl::access::fence_space::local_space);

  if (i > x4-1 || j > y4-1 || k > z4-1) { return; }

  float lap = coef0 * s_u[sui][suj][suk] + 
            coefx_1 * (s_u[sui+1][suj][suk] + s_u[sui-1][suj][suk]) +
            coefy_1 * (s_u[sui][suj+1][suk] + s_u[sui][suj-1][suk]) +
            coefz_1 * (s_u[sui][suj][suk+1] + s_u[sui][suj][suk-1]) +
            coefx_2 * (s_u[sui+2][suj][suk] + s_u[sui-2][suj][suk]) +
            coefy_2 * (s_u[sui][suj+2][suk] + s_u[sui][suj-2][suk]) +
            coefz_2 * (s_u[sui][suj][suk+2] + s_u[sui][suj][suk-2]) +
            coefx_3 * (s_u[sui+3][suj][suk] + s_u[sui-3][suj][suk]) +
            coefy_3 * (s_u[sui][suj+3][suk] + s_u[sui][suj-3][suk]) +
            coefz_3 * (s_u[sui][suj][suk+3] + s_u[sui][suj][suk-3]) +
            coefx_4 * (s_u[sui+4][suj][suk] + s_u[sui-4][suj][suk]) +
            coefy_4 * (s_u[sui][suj+4][suk] + s_u[sui][suj-4][suk]) +
            coefz_4 * (s_u[sui][suj][suk+4] + s_u[sui][suj][suk-4]);
  v[IDX3(i,j,k)] = 2.f * s_u[sui][suj][suk] + vp[IDX3(i,j,k)] * lap - v[IDX3(i,j,k)];
}

__attribute__ ((always_inline))
void target_pml_3d_kernel(
    llint nx, llint ny, llint nz, int ldimx, int ldimy, int ldimz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    float coef0,
    float coefx_1, float coefx_2, float coefx_3, float coefx_4,
    float coefy_1, float coefy_2, float coefy_3, float coefy_4,
    float coefz_1, float coefz_2, float coefz_3, float coefz_4,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    float *__restrict__ phi, const float *__restrict__ eta,
    const sycl::nd_item<3> &item, sycl::local_accessor<float, 3> s_u)
{

  const llint i0 = x3 + item.get_group(0) * item.get_local_range(0);
  const llint j0 = y3 + item.get_group(1) * item.get_local_range(1);
  const llint k0 = z3 + item.get_group(2) * item.get_local_range(2);

  const int ti = item.get_local_id(0);
  const int tj = item.get_local_id(1);
  const int tk = item.get_local_id(2);

  const llint i = i0 + ti;
  const llint j = j0 + tj;
  const llint k = k0 + tk;

  s_u[ti][tj][tk] = 0.f;

  if (ti < 2*R && tj < 2*R && tk< 2*R)
    s_u[NDIM+ti][NDIM+tj][NDIM+tk] = 0.f;

  item.barrier(sycl::access::fence_space::local_space);

  const llint sui = ti + R;
  const llint suj = tj + R;
  const llint suk = tk + R;

  const int z_side = ti / R;
  s_u[ti+z_side*NDIM][suj][suk] = u[IDX3(i+(z_side*2-1)*R,j,k)];
  const int y_side = tj / R;
  s_u[sui][tj+y_side*NDIM][suk] = u[IDX3(i,j+(y_side*2-1)*R,k)];
  s_u[sui][suj][tk] = u[IDX3(i,j,k-R)];
  s_u[sui][suj][tk+NDIM] = u[IDX3(i,j,k+R)];

  item.barrier(sycl::access::fence_space::local_space);

  if (i > x4-1 || j > y4-1 || k > z4-1) { return; }

  float lap = coef0 * s_u[sui][suj][suk] + 
            coefx_1 * (s_u[sui+1][suj][suk] + s_u[sui-1][suj][suk]) +
            coefy_1 * (s_u[sui][suj+1][suk] + s_u[sui][suj-1][suk]) +
            coefz_1 * (s_u[sui][suj][suk+1] + s_u[sui][suj][suk-1]) +
            coefx_2 * (s_u[sui+2][suj][suk] + s_u[sui-2][suj][suk]) +
            coefy_2 * (s_u[sui][suj+2][suk] + s_u[sui][suj-2][suk]) +
            coefz_2 * (s_u[sui][suj][suk+2] + s_u[sui][suj][suk-2]) +
            coefx_3 * (s_u[sui+3][suj][suk] + s_u[sui-3][suj][suk]) +
            coefy_3 * (s_u[sui][suj+3][suk] + s_u[sui][suj-3][suk]) +
            coefz_3 * (s_u[sui][suj][suk+3] + s_u[sui][suj][suk-3]) +
            coefx_4 * (s_u[sui+4][suj][suk] + s_u[sui-4][suj][suk]) +
            coefy_4 * (s_u[sui][suj+4][suk] + s_u[sui][suj-4][suk]) +
            coefz_4 * (s_u[sui][suj][suk+4] + s_u[sui][suj][suk-4]);

  const float s_eta_c = eta[IDX3(i,j,k)];

  v[IDX3(i,j,k)] = ((2.f*s_eta_c + 2.f - s_eta_c*s_eta_c)*s_u[sui][suj][suk] + 
      	    (vp[IDX3(i,j,k)] * (lap + phi[IDX3(i,j,k)]) - v[IDX3(i,j,k)])) / 
          (2.f*s_eta_c+1.f);

  phi[IDX3(i,j,k)] = 
   (phi[IDX3(i,j,k)] - 
   ((eta[IDX3(i+1,j,k)]-eta[IDX3(i-1,j,k)]) * 
   (s_u[sui+1][suj][suk]-s_u[sui-1][suj][suk]) * hdx_2 + 
   (eta[IDX3(i,j+1,k)]-eta[IDX3(i,j-1,k)]) *
   (s_u[sui][suj+1][suk]-s_u[sui][suj-1][suk]) * hdy_2 +
   (eta[IDX3(i,j,k+1)]-eta[IDX3(i,j,k-1)]) *
   (s_u[sui][suj][suk+1]-s_u[sui][suj][suk-1]) * hdz_2)) / (1.f + s_eta_c);
}

void kernel_add_source_kernel(float *g_u, llint idx, float source) {
    g_u[idx] += source;
}

void target(sycl::queue &q,
            uint nsteps, double *time_kernel,
            const struct grid_t grid,
            llint sx, llint sy, llint sz,
            float hdx_2, float hdy_2, float hdz_2,
            const float *__restrict__ coefx,
            const float *__restrict__ coefy,
            const float *__restrict__ coefz,
                  float *__restrict__ u,
            const float *__restrict__ v,
            const float *__restrict__ vp,
            const float *__restrict__ phi,
            const float *__restrict__ eta,
            const float *__restrict__ source)
{
  struct timespec start, end;

  float *d_u = allocateDeviceGrid(q, grid);
  float *d_v = allocateDeviceGrid(q, grid);
  float *d_phi = allocateDeviceGrid(q, grid);
  float *d_eta = allocateDeviceGrid(q, grid);
  float *d_vp = allocateDeviceGrid(q, grid);

  q.memset(d_u, 0, gridSize(grid));
  q.memset(d_v, 0, gridSize(grid));
  q.memcpy(d_vp, vp, gridSize(grid));
  q.memcpy(d_phi, phi, gridSize(grid));
  q.memcpy(d_eta, eta, gridSize(grid));

  const llint xmin = 0; const llint xmax = grid.nx;
  const llint ymin = 0; const llint ymax = grid.ny;

  sycl::range<3> threadsPerBlock(NDIM, NDIM, NDIM);

#ifdef DEBUG
  const uint npo = 100;
  #endif

  const float coefxyz = coefx[0] + coefy[0] + coefz[0];
  const float coefx1 = coefx[1];
  const float coefx2 = coefx[2];
  const float coefx3 = coefx[3];
  const float coefx4 = coefx[4];
  const float coefy1 = coefy[1];
  const float coefy2 = coefy[2];
  const float coefy3 = coefy[3];
  const float coefy4 = coefy[4];
  const float coefz1 = coefz[1];
  const float coefz2 = coefz[2];
  const float coefz3 = coefz[3];
  const float coefz4 = coefz[4];

  q.wait();
  clock_gettime(CLOCK_REALTIME, &start);

  for (uint istep = 1; istep <= nsteps; ++istep) {

      sycl::range<3> n_block_front((grid.nx + NDIM - 1) / NDIM,
                                   (grid.ny + NDIM - 1) / NDIM,
                                   (grid.z2 - grid.z1 + NDIM - 1) / NDIM);
      q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<float, 3> s_u_acc(
             sycl::range<3>(NDIM+2*R, NDIM+2*R, NDIM+2*R), cgh);

         cgh.parallel_for(
             sycl::nd_range<3>(n_block_front * threadsPerBlock,
                               threadsPerBlock),
             [=](sycl::nd_item<3> item) {
                target_pml_3d_kernel(
                    grid.nx, grid.ny, grid.nz, grid.ldimx, grid.ldimy,
                    grid.ldimz, xmin, xmax, ymin, ymax, grid.z1, grid.z2,
                    grid.lx, grid.ly, grid.lz, hdx_2, hdy_2, hdz_2,
                    coefxyz,
                    coefx1, coefx2, coefx3, coefx4,
                    coefy1, coefy2, coefy3, coefy4,
                    coefz1, coefz2, coefz3, coefz4,
                    d_u, d_v, d_vp, d_phi, d_eta, item, s_u_acc);
             });
      });

      sycl::range<3> n_block_top((grid.nx + NDIM - 1) / NDIM,
                                 (grid.y2 - grid.y1 + NDIM - 1) / NDIM,
                                 (grid.z4 - grid.z3 + NDIM - 1) / NDIM);
      q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<float, 3> s_u_acc(
             sycl::range<3>(NDIM+2*R, NDIM+2*R, NDIM+2*R), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(n_block_top * threadsPerBlock, threadsPerBlock),
             [=](sycl::nd_item<3> item) {
                target_pml_3d_kernel(
                    grid.nx, grid.ny, grid.nz, grid.ldimx, grid.ldimy,
                    grid.ldimz, xmin, xmax, grid.y1, grid.y2, grid.z3, grid.z4,
                    grid.lx, grid.ly, grid.lz, hdx_2, hdy_2, hdz_2,
                    coefxyz,
                    coefx1, coefx2, coefx3, coefx4,
                    coefy1, coefy2, coefy3, coefy4,
                    coefz1, coefz2, coefz3, coefz4,
                    d_u, d_v, d_vp, d_phi, d_eta, item, s_u_acc);
             });
      });

      sycl::range<3> n_block_left((grid.x2 - grid.x1 + NDIM - 1) / NDIM,
                                  (grid.y4 - grid.y3 + NDIM - 1) / NDIM,
                                  (grid.z4 - grid.z3 + NDIM - 1) / NDIM);
      q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<float, 3> s_u_acc(
             sycl::range<3>(NDIM+2*R, NDIM+2*R, NDIM+2*R), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(n_block_left * threadsPerBlock, threadsPerBlock),
             [=](sycl::nd_item<3> item) {
                target_pml_3d_kernel(
                    grid.nx, grid.ny, grid.nz, grid.ldimx, grid.ldimy,
                    grid.ldimz, grid.x1, grid.x2, grid.y3, grid.y4, grid.z3,
                    grid.z4, grid.lx, grid.ly, grid.lz, hdx_2, hdy_2, hdz_2,
                    coefxyz,
                    coefx1, coefx2, coefx3, coefx4,
                    coefy1, coefy2, coefy3, coefy4,
                    coefz1, coefz2, coefz3, coefz4,
                    d_u, d_v, d_vp, d_phi, d_eta, item, s_u_acc);
             });
      });

      sycl::range<3> n_block_center((grid.x4 - grid.x3 + NDIM - 1) / NDIM,
                                    (grid.y4 - grid.y3 + NDIM - 1) / NDIM,
                                    (grid.z4 - grid.z3 + NDIM - 1) / NDIM);
      q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<float, 3> s_u_acc(
             sycl::range<3>(NDIM+2*R, NDIM+2*R, NDIM+2*R), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(n_block_center * threadsPerBlock,
                               threadsPerBlock),
             [=](sycl::nd_item<3> item) {
                target_inner_3d_kernel(
                    grid.nx, grid.ny, grid.nz, grid.ldimx, grid.ldimy,
                    grid.ldimz, grid.x3, grid.x4, grid.y3, grid.y4, grid.z3,
                    grid.z4, grid.lx, grid.ly, grid.lz, hdx_2, hdy_2, hdz_2,
                    coefxyz,
                    coefx1, coefx2, coefx3, coefx4,
                    coefy1, coefy2, coefy3, coefy4,
                    coefz1, coefz2, coefz3, coefz4,
                    d_u, d_v, d_vp, d_phi, d_eta, item, s_u_acc);
             });
      });

      sycl::range<3> n_block_right((grid.x6 - grid.x5 + NDIM - 1) / NDIM,
                                   (grid.y4 - grid.y3 + NDIM - 1) / NDIM,
                                   (grid.z4 - grid.z3 + NDIM - 1) / NDIM);
      q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<float, 3> s_u_acc(
             sycl::range<3>(NDIM+2*R, NDIM+2*R, NDIM+2*R), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(n_block_right * threadsPerBlock,
                               threadsPerBlock),
             [=](sycl::nd_item<3> item) {
                target_pml_3d_kernel(
                    grid.nx, grid.ny, grid.nz, grid.ldimx, grid.ldimy,
                    grid.ldimz, grid.x5, grid.x6, grid.y3, grid.y4, grid.z3,
                    grid.z4, grid.lx, grid.ly, grid.lz, hdx_2, hdy_2, hdz_2,
                    coefxyz,
                    coefx1, coefx2, coefx3, coefx4,
                    coefy1, coefy2, coefy3, coefy4,
                    coefz1, coefz2, coefz3, coefz4,
                    d_u, d_v, d_vp, d_phi, d_eta, item, s_u_acc);
             });
      });

      sycl::range<3> n_block_bottom((grid.nx + NDIM - 1) / NDIM,
                                    (grid.y6 - grid.y5 + NDIM - 1) / NDIM,
                                    (grid.z4 - grid.z3 + NDIM - 1) / NDIM);
      q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<float, 3> s_u_acc(
             sycl::range<3>(NDIM+2*R, NDIM+2*R, NDIM+2*R), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(n_block_bottom * threadsPerBlock,
                               threadsPerBlock),
             [=](sycl::nd_item<3> item) {
                target_pml_3d_kernel(
                    grid.nx, grid.ny, grid.nz, grid.ldimx, grid.ldimy,
                    grid.ldimz, xmin, xmax, grid.y5, grid.y6, grid.z3, grid.z4,
                    grid.lx, grid.ly, grid.lz, hdx_2, hdy_2, hdz_2,
                    coefxyz,
                    coefx1, coefx2, coefx3, coefx4,
                    coefy1, coefy2, coefy3, coefy4,
                    coefz1, coefz2, coefz3, coefz4,
                    d_u, d_v, d_vp, d_phi, d_eta, item, s_u_acc);
             });
      });

      sycl::range<3> n_block_back((grid.nx + NDIM - 1) / NDIM,
                                  (grid.ny + NDIM - 1) / NDIM,
                                  (grid.z6 - grid.z5 + NDIM - 1) / NDIM);
      q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<float, 3> s_u_acc(
             sycl::range<3>(NDIM+2*R, NDIM+2*R, NDIM+2*R), cgh);
         cgh.parallel_for(
             sycl::nd_range<3>(n_block_back * threadsPerBlock, threadsPerBlock),
             [=](sycl::nd_item<3> item) {
                target_pml_3d_kernel(
                    grid.nx, grid.ny, grid.nz, grid.ldimx, grid.ldimy,
                    grid.ldimz, xmin, xmax, ymin, ymax, grid.z5, grid.z6,
                    grid.lx, grid.ly, grid.lz, hdx_2, hdy_2, hdz_2,
                    coefxyz,
                    coefx1, coefx2, coefx3, coefx4,
                    coefy1, coefy2, coefy3, coefy4,
                    coefz1, coefz2, coefz3, coefz4,
                    d_u, d_v, d_vp, d_phi, d_eta, item, s_u_acc);
             });
      });

      q.submit([&](sycl::handler &h) {
        float source_istep_ct2 = source[istep-1];
        h.single_task<class add_source>([=]() {
          kernel_add_source_kernel(d_v, IDX3_grid(sx, sy, sz, grid),
                                   source_istep_ct2);
        });
      });

      float *t = d_u;
      d_u = d_v;
      d_v = t;

      // Print out
      #ifdef DEBUG
      if (istep % npo == 0) printf("time step %u / %u\n", istep, nsteps);
      #endif
  }

  q.wait();
  clock_gettime(CLOCK_REALTIME, &end);
  *time_kernel = (end.tv_sec  - start.tv_sec) +
                 (double)(end.tv_nsec - start.tv_nsec) / 1.0e9;

  q.memcpy(u, d_u, gridSize(grid)).wait();

  freeDeviceGrid(q, d_u, grid);
  freeDeviceGrid(q, d_v, grid);
  freeDeviceGrid(q, d_vp, grid);
  freeDeviceGrid(q, d_phi, grid);
  freeDeviceGrid(q, d_eta, grid);
}
