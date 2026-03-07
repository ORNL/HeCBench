#include "hip/hip_runtime.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "kernel.h"
#include "reference.h"

#define max(a,b) ((a<b)?b:a)
#define min(a,b) ((a<b)?a:b)

const int NSIZE_round = NSIZE%16 ? NSIZE+16-NSIZE%16: NSIZE;
const size_t SSIZE = (size_t)NSIZE_round*48*48*48;  //Coefs size 

void  eval_abc(const float *Af, float tx, float *a) {
  a[0] = ( ( Af[0]  * tx + Af[1] ) * tx + Af[2] ) * tx + Af[3];
  a[1] = ( ( Af[4]  * tx + Af[5] ) * tx + Af[6] ) * tx + Af[7];
  a[2] = ( ( Af[8]  * tx + Af[9] ) * tx + Af[10] ) * tx + Af[11];
  a[3] = ( ( Af[12] * tx + Af[13] ) * tx + Af[14] ) * tx + Af[15];
}

int main(int argc, char ** argv) {

  float *Af = (float*) malloc (sizeof(float)*16);
  float *dAf = (float*) malloc (sizeof(float)*16);
  float *d2Af = (float*) malloc (sizeof(float)*16);

  Af[0]=-0.166667;
  Af[1]=0.500000;
  Af[2]=-0.500000;
  Af[3]=0.166667;
  Af[4]=0.500000;
  Af[5]=-1.000000;
  Af[6]=0.000000;
  Af[7]=0.666667;
  Af[8]=-0.500000;
  Af[9]=0.500000;
  Af[10]=0.500000;
  Af[11]=0.166667;
  Af[12]=0.166667;
  Af[13]=0.000000;
  Af[14]=0.000000;
  Af[15]=0.000000;
  dAf[0]=0.000000; d2Af[0]=0.000000;
  dAf[1]=-0.500000; d2Af[1]=0.000000;
  dAf[2]=1.000000; d2Af[2]=-1.000000;
  dAf[3]=-0.500000; d2Af[3]=1.000000;
  dAf[4]=0.000000; d2Af[4]=0.000000;
  dAf[5]=1.500000; d2Af[5]=0.000000;
  dAf[6]=-2.000000; d2Af[6]=3.000000;
  dAf[7]=0.000000; d2Af[7]=-2.000000;
  dAf[8]=0.000000; d2Af[8]=0.000000;
  dAf[9]=-1.500000; d2Af[9]=0.000000;
  dAf[10]=1.000000; d2Af[10]=-3.00000;
  dAf[11]=0.500000; d2Af[11]=1.000000;
  dAf[12]=0.000000; d2Af[12]=0.000000;
  dAf[13]=0.500000; d2Af[13]=0.000000;
  dAf[14]=0.000000; d2Af[14]=1.000000;
  dAf[15]=0.000000; d2Af[15]=0.000000;

  float x=0.822387;  
  float y=0.989919;  
  float z=0.104573;

  float* walkers_vals = (float*) malloc(sizeof(float)*WSIZE*NSIZE);
  float* walkers_grads = (float*) malloc(sizeof(float)*WSIZE*MSIZE);
  float* walkers_hess = (float*) malloc(sizeof(float)*WSIZE*OSIZE);
  float* walkers_vals_ref = (float*) malloc(sizeof(float)*WSIZE*NSIZE);
  float* walkers_grads_ref = (float*) malloc(sizeof(float)*WSIZE*MSIZE);
  float* walkers_hess_ref = (float*) malloc(sizeof(float)*WSIZE*OSIZE);

  float* walkers_x = (float*) malloc(sizeof(float)*WSIZE);
  float* walkers_y = (float*) malloc(sizeof(float)*WSIZE);
  float* walkers_z = (float*) malloc(sizeof(float)*WSIZE);

  for (int i=0; i<WSIZE; i++) {
    walkers_x[i] = x / WSIZE;
    walkers_y[i] = y / WSIZE;
    walkers_z[i] = z / WSIZE;
  }

  float* spline_coefs = (float*) malloc (sizeof(float)*SSIZE);
  for(size_t i=0;i<SSIZE;i++)
    spline_coefs[i]=sqrt(0.22+i*1.0)*sin(i*1.0);

  int spline_num_splines = NSIZE;
  int spline_x_grid_start = 0; 
  int spline_y_grid_start = 0; 
  int spline_z_grid_start = 0; 
  int spline_x_grid_num = 45; 
  int spline_y_grid_num = 45; 
  int spline_z_grid_num = 45; 
  int spline_x_stride=NSIZE_round*48*48;
  int spline_y_stride=NSIZE_round*48;
  int spline_z_stride=NSIZE_round;
  int spline_x_grid_delta_inv=45;
  int spline_y_grid_delta_inv=45;
  int spline_z_grid_delta_inv=45;

  float* d_walkers_vals;
  hipMalloc((void**)&d_walkers_vals, sizeof(float)*WSIZE*NSIZE);
  hipMemcpy(d_walkers_vals, walkers_vals, sizeof(float)*WSIZE*NSIZE, hipMemcpyHostToDevice);

  float* d_walkers_grads;
  hipMalloc((void**)&d_walkers_grads, sizeof(float)*WSIZE*MSIZE);
  hipMemcpy(d_walkers_grads, walkers_grads, sizeof(float)*WSIZE*MSIZE, hipMemcpyHostToDevice);

  float* d_walkers_hess;
  hipMalloc((void**)&d_walkers_hess, sizeof(float)*WSIZE*OSIZE);
  hipMemcpy(d_walkers_hess, walkers_hess, sizeof(float)*WSIZE*OSIZE, hipMemcpyHostToDevice);

  float* d_spline_coefs;
  hipMalloc((void**)&d_spline_coefs, sizeof(float)*SSIZE);
  hipMemcpy(d_spline_coefs, spline_coefs, sizeof(float)*SSIZE, hipMemcpyHostToDevice);

  float* d_a;
  hipMalloc((void**)&d_a, sizeof(float)*4);
  float* d_b;
  hipMalloc((void**)&d_b, sizeof(float)*4);
  float* d_c;
  hipMalloc((void**)&d_c, sizeof(float)*4);
  float* d_da;
  hipMalloc((void**)&d_da, sizeof(float)*4);
  float* d_db;
  hipMalloc((void**)&d_db, sizeof(float)*4);
  float* d_dc;
  hipMalloc((void**)&d_dc, sizeof(float)*4);
  float* d_d2a;
  hipMalloc((void**)&d_d2a, sizeof(float)*4);
  float* d_d2b;
  hipMalloc((void**)&d_d2b, sizeof(float)*4);
  float* d_d2c;
  hipMalloc((void**)&d_d2c, sizeof(float)*4);

  double total_time = 0.0;

  for(int i=0; i<WSIZE; i++) {
    float x = walkers_x[i], y = walkers_y[i], z = walkers_z[i];

    float ux = x*spline_x_grid_delta_inv;
    float uy = y*spline_y_grid_delta_inv;
    float uz = z*spline_z_grid_delta_inv;
    float ipartx, iparty, ipartz, tx, ty, tz;
    float a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
    intptr_t xs = spline_x_stride;
    intptr_t ys = spline_y_stride;
    intptr_t zs = spline_z_stride;

    x -= spline_x_grid_start;
    y -= spline_y_grid_start;
    z -= spline_z_grid_start;
    ipartx = (int) ux; tx = ux-ipartx; int ix = min(max(0,(int) ipartx),spline_x_grid_num-1);
    iparty = (int) uy; ty = uy-iparty; int iy = min(max(0,(int) iparty),spline_y_grid_num-1);
    ipartz = (int) uz; tz = uz-ipartz; int iz = min(max(0,(int) ipartz),spline_z_grid_num-1);

    eval_abc(Af,tx,&a[0]);
    hipMemcpy(d_a, a, sizeof(float)*4, hipMemcpyHostToDevice);

    eval_abc(Af,ty,&b[0]);
    hipMemcpy(d_b, b, sizeof(float)*4, hipMemcpyHostToDevice);

    eval_abc(Af,tz,&c[0]);
    hipMemcpy(d_c, c, sizeof(float)*4, hipMemcpyHostToDevice);

    eval_abc(dAf,tx,&da[0]);
    hipMemcpy(d_da, da, sizeof(float)*4, hipMemcpyHostToDevice);

    eval_abc(dAf,ty,&db[0]);
    hipMemcpy(d_db, db, sizeof(float)*4, hipMemcpyHostToDevice);

    eval_abc(dAf,tz,&dc[0]);
    hipMemcpy(d_dc, dc, sizeof(float)*4, hipMemcpyHostToDevice);

    eval_abc(d2Af,tx,&d2a[0]);
    hipMemcpy(d_d2a, d2a, sizeof(float)*4, hipMemcpyHostToDevice);

    eval_abc(d2Af,ty,&d2b[0]);
    hipMemcpy(d_d2b, d2b, sizeof(float)*4, hipMemcpyHostToDevice);

    eval_abc(d2Af,tz,&d2c[0]);              
    hipMemcpy(d_d2c, d2c, sizeof(float)*4, hipMemcpyHostToDevice);

    dim3 global_size((spline_num_splines+255)/256);
    dim3 local_size(256);

    bspline_ref(
        spline_coefs,
        xs, ys, zs, 
        walkers_vals_ref,
        walkers_grads_ref,
        walkers_hess_ref,
        a,
        b,
        c,
        da,
        db,
        dc,
        d2a,
        d2b,
        d2c,
        spline_x_grid_delta_inv,
        spline_y_grid_delta_inv,
        spline_z_grid_delta_inv,
        spline_num_splines,
        i, ix, iy, iz	);

    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    bspline<<<global_size, local_size>>>(
        d_spline_coefs,
        xs, ys, zs, 
        d_walkers_vals,
        d_walkers_grads,
        d_walkers_hess,
        d_a,
        d_b,
        d_c,
        d_da,
        d_db,
        d_dc,
        d_d2a,
        d_d2b,
        d_d2c,
        spline_x_grid_delta_inv,
        spline_y_grid_delta_inv,
        spline_z_grid_delta_inv,
        spline_num_splines,
        i, ix, iy, iz	);

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }
  printf("Total kernel execution time %lf (s)\n", total_time * 1e-9);

  hipMemcpy(walkers_vals, d_walkers_vals, sizeof(float)*WSIZE*NSIZE, hipMemcpyDeviceToHost);
  hipMemcpy(walkers_grads, d_walkers_grads, sizeof(float)*WSIZE*MSIZE, hipMemcpyDeviceToHost);
  hipMemcpy(walkers_hess, d_walkers_hess, sizeof(float)*WSIZE*OSIZE, hipMemcpyDeviceToHost);

  bool ok = true;
  const float atol = 2.f;
  for( int i = 0; i < WSIZE * NSIZE; i++ ) {
    if (fabsf(walkers_vals[i] - walkers_vals_ref[i]) > atol) {
      //printf("%f %f\n", walkers_vals[i] , walkers_vals_ref[i]);
      ok = false;
      break;
    }
  }
  for( int i = 0; i < WSIZE * MSIZE; i++ ) {
    if (fabsf(walkers_grads[i] - walkers_grads_ref[i]) > atol) {
      //printf("%f %f\n", walkers_grads[i] , walkers_grads_ref[i]);
      ok = false;
      break;
    }
  }
  for( int i = 0; i < WSIZE * OSIZE; i++ ) {
    if (fabsf(walkers_hess[i] - walkers_hess_ref[i]) > atol) {
      //printf("%f %f\n", walkers_hess[i] , walkers_hess_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(Af);
  free(dAf);
  free(d2Af);
  free(walkers_vals);
  free(walkers_grads);
  free(walkers_hess);
  free(walkers_vals_ref);
  free(walkers_grads_ref);
  free(walkers_hess_ref);
  free(walkers_x);
  free(walkers_y);
  free(walkers_z);
  free(spline_coefs);

  hipFree(d_walkers_vals);
  hipFree(d_walkers_grads);
  hipFree(d_walkers_hess);
  hipFree(d_spline_coefs);
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);
  hipFree(d_da);
  hipFree(d_db);
  hipFree(d_dc);
  hipFree(d_d2a);
  hipFree(d_d2b);
  hipFree(d_d2c);
  return 0;
}
