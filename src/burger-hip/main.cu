#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include "kernels.h"

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <dim_x> <dim_y> <nt>\n", argv[0]);
    printf("dim_x: number of grid points in the x axis\n");
    printf("dim_y: number of grid points in the y axis\n");
    printf("nt: number of time steps\n");
    exit(-1);
  }

  // Define the domain
  const int x_points = atoi(argv[1]);
  const int y_points = atoi(argv[2]);
  const int num_itrs = atoi(argv[3]);
  const double x_len = 2.0;
  const double y_len = 2.0;
  const double del_x = x_len/(x_points-1);
  const double del_y = y_len/(y_points-1);

  const int grid_elems = x_points * y_points;
  const int grid_size = sizeof(double) * grid_elems;

  double *x = (double*) malloc (sizeof(double) * x_points);
  double *y = (double*) malloc (sizeof(double) * y_points);
  double *u = (double*) malloc (grid_size);
  double *v = (double*) malloc (grid_size);
  double *u_new = (double*) malloc (grid_size);
  double *v_new = (double*) malloc (grid_size);

  // store device results
  double *du = (double*) malloc (grid_size);
  double *dv = (double*) malloc (grid_size);

  // Define the parameters
  const double nu = 0.01;
  const double sigma = 0.0009;
  const double del_t = sigma * del_x * del_y / nu;      // CFL criteria

  printf("2D Burger's equation\n");
  printf("Grid dimension: x = %d y = %d\n", x_points, y_points);
  printf("Number of time steps: %d\n", num_itrs);

  for(int i = 0; i < x_points; i++) x[i] = i * del_x;
  for(int i = 0; i < y_points; i++) y[i] = i * del_y;

  for(int i = 0; i < y_points; i++){
    for(int j = 0; j < x_points; j++){
      u[idx(i,j)] = 1.0;
      v[idx(i,j)] = 1.0;
      u_new[idx(i,j)] = 1.0;
      v_new[idx(i,j)] = 1.0;

      if(x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0){
        u[idx(i,j)] = 2.0;
        v[idx(i,j)] = 2.0;
        u_new[idx(i,j)] = 2.0;
        v_new[idx(i,j)] = 2.0;
      }
    }
  }

  double *d_u_new;
  hipMalloc((void**)&d_u_new, grid_size);

  double *d_v_new;
  hipMalloc((void**)&d_v_new, grid_size);

  double *d_u;
  hipMalloc((void**)&d_u, grid_size);

  double *d_v;
  hipMalloc((void**)&d_v, grid_size);

  hipMemcpy(d_u_new, u_new, grid_size, hipMemcpyHostToDevice);
  hipMemcpy(d_v_new, v_new, grid_size, hipMemcpyHostToDevice);
  hipMemcpy(d_u, u, grid_size, hipMemcpyHostToDevice);
  hipMemcpy(d_v, v, grid_size, hipMemcpyHostToDevice);

  // ranges of the four kernels
  dim3 grid ((x_points-2+15)/16, (y_points-2+15)/16);
  dim3 block (16, 16);
  dim3 grid2 ((x_points+255)/256);
  dim3 block2 (256);
  dim3 grid3 ((y_points+255)/256);
  dim3 block3 (256);
  dim3 grid4 ((grid_elems+255)/256);
  dim3 block4 (256);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for(int itr = 0; itr < num_itrs; itr++){

    hipLaunchKernelGGL(core, dim3(grid), dim3(block), 0, 0, d_u_new, d_v_new, d_u, d_v, x_points, y_points, nu, del_t, del_x, del_y);

    // Boundary conditions
    hipLaunchKernelGGL(bound_h, dim3(grid2), dim3(block2), 0, 0, d_u_new, d_v_new, x_points, y_points);

    hipLaunchKernelGGL(bound_v, dim3(grid3), dim3(block3), 0, 0, d_u_new, d_v_new, x_points, y_points);

    // Updating older values to newer ones
    hipLaunchKernelGGL(update, dim3(grid4), dim3(block4), 0, 0, d_u, d_v, d_u_new, d_v_new, grid_elems);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time %f (s)\n", time * 1e-9f);

  hipMemcpy(du, d_u, grid_size, hipMemcpyDeviceToHost);
  hipMemcpy(dv, d_v, grid_size, hipMemcpyDeviceToHost);

  printf("Serial computing for verification...\n");

  // Reset velocities
  for(int i = 0; i < y_points; i++){
    for(int j = 0; j < x_points; j++){
      u[idx(i,j)] = 1.0;
      v[idx(i,j)] = 1.0;
      u_new[idx(i,j)] = 1.0;
      v_new[idx(i,j)] = 1.0;

      if(x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0){
        u[idx(i,j)] = 2.0;
        v[idx(i,j)] = 2.0;
        u_new[idx(i,j)] = 2.0;
        v_new[idx(i,j)] = 2.0;
      }
    }
  }

  for(int itr = 0; itr < num_itrs; itr++){

    for(int i = 1; i < y_points-1; i++){
      for(int j = 1; j < x_points-1; j++){
        u_new[idx(i,j)] = u[idx(i,j)] + (nu*del_t/(del_x*del_x)) * (u[idx(i,j+1)] + u[idx(i,j-1)] - 2 * u[idx(i,j)]) + 
          (nu*del_t/(del_y*del_y)) * (u[idx(i+1,j)] + u[idx(i-1,j)] - 2 * u[idx(i,j)]) - 
          (del_t/del_x)*u[idx(i,j)] * (u[idx(i,j)] - u[idx(i,j-1)]) - 
          (del_t/del_y)*v[idx(i,j)] * (u[idx(i,j)] - u[idx(i-1,j)]);

        v_new[idx(i,j)] = v[idx(i,j)] + (nu*del_t/(del_x*del_x)) * (v[idx(i,j+1)] + v[idx(i,j-1)] - 2 * v[idx(i,j)]) + 
          (nu*del_t/(del_y*del_y)) * (v[idx(i+1,j)] + v[idx(i-1,j)] - 2 * v[idx(i,j)]) -
          (del_t/del_x)*u[idx(i,j)] * (v[idx(i,j)] - v[idx(i,j-1)]) - 
          (del_t/del_y)*v[idx(i,j)] * (v[idx(i,j)] - v[idx(i-1,j)]);
      }
    }

    // Boundary conditions
    for(int i = 0; i < x_points; i++){
      u_new[idx(0,i)] = 1.0;
      v_new[idx(0,i)] = 1.0;
      u_new[idx(y_points-1,i)] = 1.0;
      v_new[idx(y_points-1,i)] = 1.0;
    }

    for(int j = 0; j < y_points; j++){
      u_new[idx(j,0)] = 1.0;
      v_new[idx(j,0)] = 1.0;
      u_new[idx(j,x_points-1)] = 1.0;
      v_new[idx(j,x_points-1)] = 1.0;
    }

    // Updating older values to newer ones
    for(int i = 0; i < y_points; i++){
      for(int j = 0; j < x_points; j++){
        u[idx(i,j)] = u_new[idx(i,j)];
        v[idx(i,j)] = v_new[idx(i,j)];
      }
    }
  }

  bool ok = true;
  for(int i = 0; i < y_points; i++){
    for(int j = 0; j < x_points; j++){
      if (fabs(du[idx(i,j)] - u[idx(i,j)]) > 1e-6 || 
          fabs(dv[idx(i,j)] - v[idx(i,j)]) > 1e-6) ok = false;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(x);
  free(y);
  free(u);
  free(v);
  free(du);
  free(dv);
  free(u_new);
  free(v_new);
  hipFree(d_u);
  hipFree(d_v);
  hipFree(d_u_new);
  hipFree(d_v_new);

  return 0;
}
