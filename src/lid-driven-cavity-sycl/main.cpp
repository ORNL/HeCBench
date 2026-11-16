/** GPU solver for 2D lid-driven cavity problem, using finite difference method
 * \file main_gpu.cpp
 *
 * Solve the incompressible, isothermal 2D Navier–Stokes equations for a square
 * lid-driven cavity on a GPU (via CUDA), using the finite difference method.
 * To change the grid resolution, modify "NUM". In addition, the problem is controlled
 * by the Reynolds number ("Re_num").
 *
 * Based on the methodology given in Chapter 3 of "Numerical Simulation in Fluid
 * Dynamics", by M. Griebel, T. Dornseifer, and T. Neunhoeffer. SIAM, Philadelphia,
 * PA, 1998.
 *
 * Boundary conditions:
 * u = 0 and v = 0 at x = 0, x = L, y = 0
 * u = ustar at y = H
 * v = 0 at y = H
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "constants.h"
#include "../lid-driven-cavity-cuda/reference.h"

/** Mesh sizes */
const Real dx = xLength / NUM;
const Real dy = yLength / NUM;

/** Max macro (type safe, from GNU) */
//#define MAX(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })

/** Min macro (type safe) */
//#define MIN(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

// map two-dimensional indices to one-dimensional indices for device memory
#define u(I, J) u_d[((I) * ((NUM) + 2)) + (J)]
#define v(I, J) v_d[((I) * ((NUM) + 2)) + (J)]
#define F(I, J) F_d[((I) * ((NUM) + 2)) + (J)]
#define G(I, J) G_d[((I) * ((NUM) + 2)) + (J)]
#define pres_red(I, J) pres_red_d[((I) * ((NUM_2) + 2)) + (J)]
#define pres_black(I, J) pres_black_d[((I) * ((NUM_2) + 2)) + (J)]

///////////////////////////////////////////////////////////////////////////////
void set_BCs_host (Real*, Real*, Real&, Real&);

int main (int argc, char *argv[])
{
  // iterations for Red-Black Gauss-Seidel with SOR
  int iter = 0;

  const int it_max = 1000000;

  // SOR iteration tolerance
  const Real tol = 0.001;

  // time range
  const Real time_start = 0.0;
  const Real time_end = 0.001; //20.0;

  // initial time step size
  Real dt = 0.02;

  int size = (NUM + 2) * (NUM + 2);
  int size_pres = ((NUM / 2) + 2) * (NUM + 2);

  // arrays for pressure and velocity
  Real* F;
  Real* u;
  Real* G;
  Real* v;

  F = (Real *) calloc (size, sizeof(Real));
  u = (Real *) calloc (size, sizeof(Real));
  G = (Real *) calloc (size, sizeof(Real));
  v = (Real *) calloc (size, sizeof(Real));

  for (int i = 0; i < size; ++i) {
    F[i] = ZERO;
    u[i] = ZERO;
    G[i] = ZERO;
    v[i] = ZERO;
  }

  // arrays for pressure
  Real* pres_red;
  Real* pres_black;

  pres_red = (Real *) calloc (size_pres, sizeof(Real));
  pres_black = (Real *) calloc (size_pres, sizeof(Real));

  for (int i = 0; i < size_pres; ++i) {
    pres_red[i] = ZERO;
    pres_black[i] = ZERO;
  }

  // print problem size
  printf("Problem size: %d x %d \n", NUM, NUM);

  ////////////////////////////////////////
  // block and grid dimensions

  // boundary conditions kernel
  sycl::range<1> lws_bcs (BLOCK_SIZE);
  sycl::range<1> gws_bcs (NUM);

  // pressure kernel
  sycl::range<2> lws_pr (1, BLOCK_SIZE);
  sycl::range<2> gws_pr (NUM, NUM / 2);

  // block and grid dimensions for F
  sycl::range<2> lws_F (1, BLOCK_SIZE);
  sycl::range<2> gws_F (NUM, NUM);

  // block and grid dimensions for G
  sycl::range<2> lws_G (1, BLOCK_SIZE);
  sycl::range<2> gws_G (NUM, NUM);

  // horizontal pressure boundary conditions
  sycl::range<1> lws_hpbc (BLOCK_SIZE);
  sycl::range<1> gws_hpbc (NUM / 2);

  // vertical pressure boundary conditions
  sycl::range<1> lws_vpbc (BLOCK_SIZE);
  sycl::range<1> gws_vpbc (NUM / 2);
  ///////////////////////////////////////////

  // residual variable
  Real* res;

  int size_res = NUM / (2 * BLOCK_SIZE) * NUM;
  res = (Real *) calloc (size_res, sizeof(Real));

  // variables to store maximum velocities
  Real* max_u_arr;
  Real* max_v_arr;
  int size_max = size_res;

  max_u_arr = (Real *) calloc (size_max, sizeof(Real));
  max_v_arr = (Real *) calloc (size_max, sizeof(Real));

  // pressure sum
  Real* pres_sum;
  pres_sum = (Real *) calloc (size_res, sizeof(Real));

  // set initial BCs
  Real max_u = SMALL;
  Real max_v = SMALL;

  set_BCs_host (u, v, max_u, max_v);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    // allocate and transfer device memory
  Real *u_d = sycl::malloc_device<Real>(size, q);
  Real *v_d = sycl::malloc_device<Real>(size, q);
  Real *F_d = sycl::malloc_device<Real>(size, q);
  Real *G_d = sycl::malloc_device<Real>(size, q);
  Real *pres_red_d = sycl::malloc_device<Real>(size_pres, q);
  Real *pres_black_d = sycl::malloc_device<Real>(size_pres, q);

  q.memcpy (u_d, u, size * sizeof(Real));
  q.memcpy (F_d, F, size * sizeof(Real));
  q.memcpy (v_d, v, size * sizeof(Real));
  q.memcpy (G_d, G, size * sizeof(Real));
  q.memcpy (pres_red_d, pres_red, size_pres * sizeof(Real));
  q.memcpy (pres_black_d, pres_black, size_pres * sizeof(Real));

  Real *pres_sum_d = sycl::malloc_device<Real>(size_res, q);
  Real *res_d = sycl::malloc_device<Real>(size_res, q);
  Real *max_u_d = sycl::malloc_device<Real>(size_max, q);
  Real *max_v_d = sycl::malloc_device<Real>(size_max, q);

  // state at the start of the last time step, for verification
  Real *u0_d = sycl::malloc_device<Real>(size, q);
  Real *v0_d = sycl::malloc_device<Real>(size, q);
  Real *pres_red0_d = sycl::malloc_device<Real>(size_pres, q);
  Real *pres_black0_d = sycl::malloc_device<Real>(size_pres, q);

  Real time = time_start;

  // time-step size based on grid and Reynolds number
  Real dt_Re = 0.5 * Re_num / ((1.0 / (dx * dx)) + (1.0 / (dy * dy)));

  q.wait();
  auto start = std::chrono::steady_clock::now();

  // time iteration loop
  while (time < time_end) {

    // calculate time step based on stability and CFL
    dt = FMIN((dx / max_u), (dy / max_v));
    dt = tau * FMIN(dt_Re, dt);

    if ((time + dt) >= time_end) {
      dt = time_end - time;

      // snapshot the state for verify_last_step; the final step always takes this branch
      q.memcpy (u0_d, u_d, size * sizeof(Real));
      q.memcpy (v0_d, v_d, size * sizeof(Real));
      q.memcpy (pres_red0_d, pres_red_d, size_pres * sizeof(Real));
      q.memcpy (pres_black0_d, pres_black_d, size_pres * sizeof(Real));
    }

    // calculate F and G
    //calculate_F <<<grid_F, block_F>>> (dt, u_d, v_d, F_d);
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class calculate_F>(
        sycl::nd_range<2>(gws_F, lws_F), [=] (sycl::nd_item<2> item) {
        #include "calculate_F.sycl"
      });
    });

    //    calculate_G <<<grid_G, block_G>>> (dt, u_d, v_d, G_d);
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class calculate_G>(
        sycl::nd_range<2>(gws_G, lws_G), [=] (sycl::nd_item<2> item) {
        #include "calculate_G.sycl"
      });
    });

    // get L2 norm of initial pressure
    //sum_pressure <<<grid_pr, block_pr>>> (pres_red_d, pres_black_d, pres_sum_d);
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor <Real, 1> sum_cache (sycl::range<1>(BLOCK_SIZE), h);
      h.parallel_for<class sum_pressure>(
        sycl::nd_range<2>(gws_pr, lws_pr), [=] (sycl::nd_item<2> item) {
        #include "sum_pressure.sycl"
      });
    });

    q.memcpy (pres_sum, pres_sum_d, size_res * sizeof(Real)).wait();

    Real p0_norm = ZERO;
    #pragma unroll
    for (int i = 0; i < size_res; ++i) {
      p0_norm += pres_sum[i];
    }
    //printf("p0_norm = %lf\n", p0_norm);

    p0_norm = SQRT(p0_norm / ((Real)(NUM * NUM)));
    if (p0_norm < 0.0001) {
      p0_norm = 1.0;
    }

    Real norm_L2;

    // calculate new pressure
    // red-black Gauss-Seidel with SOR iteration loop
    for (iter = 1; iter <= it_max; ++iter) {

      // set pressure boundary conditions
      //set_horz_pres_BCs <<<grid_hpbc, block_hpbc>>> (pres_red_d, pres_black_d);
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class set_horz_pres_BCs>(
          sycl::nd_range<1>(gws_hpbc, lws_hpbc), [=] (sycl::nd_item<1> item) {
          #include "set_horz_pres_BCs.sycl"
        });
      });

      //      set_vert_pres_BCs <<<grid_vpbc, block_hpbc>>> (pres_red_d, pres_black_d);
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class set_vert_pres_BCs>(
          sycl::nd_range<1>(gws_vpbc, lws_vpbc), [=] (sycl::nd_item<1> item) {
          #include "set_vert_pres_BCs.sycl"
        });
      });

      // update red cells
      //      red_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_black_d, pres_red_d);
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class red_kernel>(
          sycl::nd_range<2>(gws_pr, lws_pr), [=] (sycl::nd_item<2> item) {
          #include "red_kernel.sycl"
        });
      });

      // update black cells
      //      black_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_red_d, pres_black_d);
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class black_kernel>(
          sycl::nd_range<2>(gws_pr, lws_pr), [=] (sycl::nd_item<2> item) {
          #include "black_kernel.sycl"
        });
      });

      // calculate residual values
      //calc_residual <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_red_d, pres_black_d, res_d);
      q.submit([&](sycl::handler &h) {
        sycl::local_accessor <Real, 1> sum_cache (sycl::range<1>(BLOCK_SIZE), h);
        h.parallel_for<class calc_residual>(
          sycl::nd_range<2>(gws_pr, lws_pr), [=] (sycl::nd_item<2> item) {
          #include "calc_residual.sycl"
        });
      });

      // transfer residual value(s) back to CPU
      q.memcpy (res, res_d, size_res * sizeof(Real)).wait();

      norm_L2 = ZERO;
      #pragma unroll
      for (int i = 0; i < size_res; ++i) {
        norm_L2 += res[i];
      }

    //printf("norm_L2 = %lf\n", norm_L2);
      // calculate residual
      norm_L2 = SQRT(norm_L2 / ((Real)(NUM * NUM))) / p0_norm;

      // if tolerance has been reached, end SOR iterations
      if (norm_L2 < tol) {
        break;
      }
    } // end for

    printf("Time = %f, delt = %e, iter = %i, res = %e\n", time + dt, dt, iter, norm_L2);

    // calculate new velocities and transfer maximums back

    //calculate_u <<<grid_pr, block_pr>>> (dt, F_d, pres_red_d, pres_black_d, u_d, max_u_d);
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor <Real, 1> max_cache (sycl::range<1>(BLOCK_SIZE), h);
      h.parallel_for<class calculate_u>(
        sycl::nd_range<2>(gws_pr, lws_pr), [=] (sycl::nd_item<2> item) {
        #include "calculate_u.sycl"
      });
    });

    q.memcpy (max_u_arr, max_u_d, size_max * sizeof(Real));

    //    calculate_v <<<grid_pr, block_pr>>> (dt, G_d, pres_red_d, pres_black_d, v_d, max_v_d);
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor <Real, 1> max_cache (sycl::range<1>(BLOCK_SIZE), h);
      h.parallel_for<class calculate_v>(
        sycl::nd_range<2>(gws_pr, lws_pr), [=] (sycl::nd_item<2> item) {
        #include "calculate_v.sycl"
      });
    });

    q.memcpy (max_v_arr, max_v_d, size_max * sizeof(Real)).wait();

    // get maximum u- and v- velocities
    max_v = SMALL;
    max_u = SMALL;

    #pragma unroll
    for (int i = 0; i < size_max; ++i) {
      Real test_u = max_u_arr[i];
      max_u = FMAX(max_u, test_u);

      Real test_v = max_v_arr[i];
      max_v = FMAX(max_v, test_v);
    }

    // set velocity boundary conditions
    //set_BCs <<<grid_bcs, block_bcs>>> (u_d, v_d);
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class set_BCs>(
        sycl::nd_range<1>(gws_bcs, lws_bcs), [=] (sycl::nd_item<1> item) {
        #include "set_BCs.sycl"
      });
    });

    // increase time
    time += dt;

    // single time step
    //break;

  } // end while

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nTotal execution time of the iteration loop: %f (s)\n", elapsed_time * 1e-9f);
  
  // transfer final temperature values back implicitly
  q.memcpy (u, u_d, size * sizeof(Real));
  q.memcpy (v, v_d, size * sizeof(Real));
  q.memcpy (pres_red, pres_red_d, size_pres * sizeof(Real));
  q.memcpy (pres_black, pres_black_d, size_pres * sizeof(Real));
  q.memcpy (F, F_d, size * sizeof(Real));
  q.memcpy (G, G_d, size * sizeof(Real));

  Real* u0 = (Real *) malloc (size * sizeof(Real));
  Real* v0 = (Real *) malloc (size * sizeof(Real));
  Real* pres_red0 = (Real *) malloc (size_pres * sizeof(Real));
  Real* pres_black0 = (Real *) malloc (size_pres * sizeof(Real));
  q.memcpy (u0, u0_d, size * sizeof(Real));
  q.memcpy (v0, v0_d, size * sizeof(Real));
  q.memcpy (pres_red0, pres_red0_d, size_pres * sizeof(Real));
  q.memcpy (pres_black0, pres_black0_d, size_pres * sizeof(Real));
  q.wait();

  bool ok = verify_last_step<Real>(NUM, dx, dy, dt, Re_num, mix_param, gx, gy, tol,
                                   u0, v0, pres_red0, pres_black0,
                                   F, G, pres_red, pres_black, u, v);
  printf("%s\n", ok ? "PASS" : "FAIL");

  // write data to file
  FILE * pfile;
  pfile = fopen("velocity_gpu.dat", "w");
  fprintf(pfile, "#x\ty\tu\tv\n");
  if (pfile != NULL) {
    for (int row = 0; row < NUM; ++row) {
      for (int col = 0; col < NUM; ++col) {

        Real u_ij = u[(col * NUM) + row];
        Real u_im1j;
        if (col == 0) {
          u_im1j = 0.0;
        } else {
          u_im1j = u[(col - 1) * NUM + row];
        }

        u_ij = (u_ij + u_im1j) / 2.0;

        Real v_ij = v[(col * NUM) + row];
        Real v_ijm1;
        if (row == 0) {
          v_ijm1 = 0.0;
        } else {
          v_ijm1 = v[(col * NUM) + row - 1];
        }

        v_ij = (v_ij + v_ijm1) / 2.0;

        fprintf(pfile, "%f\t%f\t%f\t%f\n", ((Real)col + 0.5) * dx, ((Real)row + 0.5) * dy, u_ij, v_ij);
      }
    }
  }

  fclose(pfile);

  // free device memory
  sycl::free(u_d, q);
  sycl::free(v_d, q);
  sycl::free(F_d, q);
  sycl::free(G_d, q);
  sycl::free(pres_red_d, q);
  sycl::free(pres_black_d, q);
  sycl::free(max_u_d, q);
  sycl::free(max_v_d, q);
  sycl::free(pres_sum_d, q);
  sycl::free(res_d, q);
  sycl::free(u0_d, q);
  sycl::free(v0_d, q);
  sycl::free(pres_red0_d, q);
  sycl::free(pres_black0_d, q);

  free(u0);
  free(v0);
  free(pres_red0);
  free(pres_black0);

  free(pres_red);
  free(pres_black);
  free(u);
  free(v);
  free(F);
  free(G);
  free(max_u_arr);
  free(max_v_arr);
  free(res);
  free(pres_sum);
  return ok ? 0 : 1;
}
