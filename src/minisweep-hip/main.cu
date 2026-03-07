#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include "utils.h"
#include "utils.cu"
#include "kernels.cu"

/*===========================================================================*/
/*---Main---*/

int main( int argc, char** argv )
{
  Arguments args;
  memset( (void*)&args, 0, sizeof(Arguments) );

  args.argc = argc;
  args.argv_unconsumed = (char**) malloc( argc * sizeof( char* ) );
  args.argstring = 0;

  for( int i=0; i<argc; ++i )
  {
    if ( argv[i] == NULL) {
      printf("Null command line argument encountered");
      return -1;
    }
    args.argv_unconsumed[i] = argv[i];
  }

  Dimensions dims_g;       /*---dims for entire problem---*/
  Dimensions dims;         /*---dims for the part on this MPI proc---*/

  Sweeper sweeper ;
  memset( (void*)&sweeper, 0, sizeof(Sweeper) );

  int niterations = 0;

  /*---Define problem specs---*/
  dims_g.ncell_x = Arguments_consume_int_or_default( &args, "--ncell_x",  5 );
  dims_g.ncell_y = Arguments_consume_int_or_default( &args, "--ncell_y",  5 );
  dims_g.ncell_z = Arguments_consume_int_or_default( &args, "--ncell_z",  5 );
  dims_g.ne   = Arguments_consume_int_or_default( &args, "--ne", 30 );
  dims_g.na   = Arguments_consume_int_or_default( &args, "--na", 33 );
  niterations = Arguments_consume_int_or_default( &args, "--niterations", 1 );
  dims_g.nm   = NM;

  if (dims_g.ncell_x <= 0) { printf("Invalid ncell_x supplied."); return -1; }
  if (dims_g.ncell_y <= 0) { printf("Invalid ncell_y supplied."); return -1; }
  if (dims_g.ncell_z <= 0) { printf("Invalid ncell_z supplied."); return -1; }
  if (dims_g.ne <= 0     ) { printf("Invalid ne supplied."); return -1; }
  if (dims_g.nm <= 0     ) { printf("Invalid nm supplied."); return -1; }
  if (dims_g.na <= 0     ) { printf("Invalid na supplied."); return -1; }
  if (niterations < 1    ) { printf("Invalid iteration count supplied."); return -1; }

  /*---Initialize (local) dimensions - no domain decomposition---*/
  dims.ncell_x = dims_g.ncell_x;
  dims.ncell_y = dims_g.ncell_y;
  dims.ncell_z = dims_g.ncell_z;
  dims.ne = dims_g.ne;
  dims.nm = dims_g.nm;
  dims.na = dims_g.na;

  /*---Initialize quantities---*/
  size_t n = dims.nm * dims.na * NOCTANT * sizeof(P);
  P* a_from_m = (P*) malloc (n);
  P* m_from_a = (P*) malloc (n);

  /*---First set to zero---*/
  for( int octant=0; octant<NOCTANT; ++octant )
    for( int im=0;     im<dims.nm;     ++im )
      for( int ia=0;     ia<dims.na;     ++ia )
        a_from_m[A_FROM_M_ADDR(dims.na, im, ia, octant)]  = (P)0;

  for( int octant=0; octant<NOCTANT; ++octant )
    for( int i=0;      i<dims.na;      ++i )
    {
      const int quot = ( i + 1 ) / dims.nm;
      const int rem  = ( i + 1 ) % dims.nm;
      a_from_m[A_FROM_M_ADDR(dims.na, dims.nm-1, i, octant)] += quot;

      if( rem != 0 )
      {
        a_from_m[A_FROM_M_ADDR(dims.na, 0, i, octant)] += (P)-1;
        a_from_m[A_FROM_M_ADDR(dims.na, rem, i, octant)] += (P)1;
      }
    }

  /*---Fill matrix with entries that leave linears unaffected---*/

  /*---This is to create a more dense, nontrivial matrix, with additions
    to the rows that are guaranteed to send affine functions to zero.
    ---*/

  for(int octant=0; octant<NOCTANT; ++octant )
    for(int im=0;     im<dims.nm-2;   ++im )
      for(int ia=0;     ia<dims.na;     ++ia )
      {
        const int randvalue = 21 + ( im + dims.nm * ia ) % 17;
        a_from_m[A_FROM_M_ADDR(dims.na, im, ia, octant)] += -randvalue;
        a_from_m[A_FROM_M_ADDR(dims.na, im+1, ia, octant)] += 2*randvalue;
        a_from_m[A_FROM_M_ADDR(dims.na, im+2, ia, octant)] += -randvalue;
      }
#ifdef DEBUG
  for (int i = 0; i < n/sizeof(P); i++) printf("a_from_m %d %f\n", i, a_from_m[i]);
#endif

  P* d_a_from_m = NULL; 
  hipMalloc((void**) &d_a_from_m, n);
  hipMemcpy(d_a_from_m, a_from_m, n, hipMemcpyHostToDevice); 

  // m from a
  for( int octant=0; octant<NOCTANT; ++octant )
    for( int im=0;     im<dims.nm;     ++im )
      for( int ia=0;     ia<dims.na;     ++ia )
        m_from_a[M_FROM_A_ADDR(dims.na, im, ia, octant)]  = (P)0;

  for( int octant=0; octant<NOCTANT; ++octant )
    for( int i=0;      i<dims.nm;      ++i )
    {
      const int quot = ( i + 1 ) / dims.na;
      const int rem  = ( i + 1 ) % dims.na;
      m_from_a[M_FROM_A_ADDR(dims.na, i, dims.na-1, octant)] += quot;

      if( rem != 0 )
      {
        m_from_a[M_FROM_A_ADDR(dims.na, i, 0, octant)] += (P)-1;
        m_from_a[M_FROM_A_ADDR(dims.na, i, rem, octant)] += (P)1;
      }
    }

  /*---Fill matrix with entries that leave linears unaffected---*/

  /*---This is to create a more dense, nontrivial matrix, with additions
    to the rows that are guaranteed to send affine functions to zero.
    ---*/

  for(int octant=0; octant<NOCTANT; ++octant )
    for(int im=0;     im<dims.nm;   ++im )
      for(int ia=0;     ia<dims.na-2;     ++ia )
      {
        const int randvalue = 37 + ( im + dims.nm * ia ) % 19;
        m_from_a[M_FROM_A_ADDR(dims.na, im, ia, octant)] += -randvalue;
        m_from_a[M_FROM_A_ADDR(dims.na, im, ia+1, octant)] += 2*randvalue;
        m_from_a[M_FROM_A_ADDR(dims.na, im, ia+2, octant)] += -randvalue;
      }
  /*---Scale matrix to compensate for 8 octants and also angle scale factor---*/
  for(int octant=0; octant<NOCTANT; ++octant )
    for(int im=0;     im<dims.nm;     ++im )
      for(int ia=0;     ia<dims.na;     ++ia )
      {
        m_from_a[M_FROM_A_ADDR(dims.na, im, ia, octant)] /= NOCTANT;
        // scale factor angle
        m_from_a[M_FROM_A_ADDR(dims.na, im, ia, octant)] /= 1 << ( ia & ( (1<<3) - 1 ) ); 
      }
#ifdef DEBUG
  for (int i = 0; i < n/sizeof(P); i++) printf("m_from_a %d %f\n", i, m_from_a[i]);
#endif


  P* d_m_from_a = NULL; 
  hipMalloc((void**) &d_m_from_a, n);
  hipMemcpy(d_m_from_a, m_from_a, n, hipMemcpyHostToDevice); 

  n = Dimensions_size_state( dims, NU ) * sizeof(P);
  P* vi = (P*) malloc( n );

  /*---Initialize input state array ---*/
  initialize_input_state( vi, dims, NU );

#ifdef DEBUG
  for (int i = 0; i < n/sizeof(P); i++) printf("vi %d %f\n", i, vi[i]);
#endif

  P* d_vi = NULL;
  hipMalloc((void**) &d_vi, n);
  hipMemcpy(d_vi, vi, n, hipMemcpyHostToDevice); 

  P* vo = (P*) malloc( n );
  P* d_vo = NULL;
  hipMalloc((void**) &d_vo, n );


  /*---This is not strictly required for the output state array, but might
    have a performance effect from pre-touching pages */
  //for (int i = 0; i < Dimensions_size_state( dims, NU ); i++) vo[i] = (P)0;

  /*---Initialize sweeper---*/
  sweeper.nblock_z = 1; //NOTE: will not work efficiently in parallel.
  sweeper.noctant_per_block = NOCTANT;
  sweeper.nblock_octant     = 1;

  sweeper.dims = dims;
  sweeper.dims_b = dims;
  sweeper.dims_b.ncell_z = dims.ncell_z / sweeper.nblock_z;

  // step scheduler
  sweeper.stepscheduler.nblock_z_          = sweeper.nblock_z;
  sweeper.stepscheduler.nproc_x_           = 1; //Env_nproc_x( env );
  sweeper.stepscheduler.nproc_y_           = 1; //Env_nproc_y( env );
  sweeper.stepscheduler.nblock_octant_     = sweeper.nblock_octant;
  sweeper.stepscheduler.noctant_per_block_ = NOCTANT / sweeper.nblock_octant;
  //sweeper.faces.noctant_per_block_         = sweeper.noctant_per_block;

  const int noctant_per_block = sweeper.noctant_per_block; 

  n = Dimensions_size_facexy( sweeper.dims_b, NU, noctant_per_block ) * sizeof(P);
  P* facexy = (P*) malloc ( n );
  P* d_facexy = NULL;
  hipMalloc((void**)&d_facexy, n);

  n = Dimensions_size_facexz( sweeper.dims_b, NU, noctant_per_block) * sizeof(P);

  P* facexz = (P*) malloc ( n );
  P* d_facexz = NULL;
  hipMalloc((void**) &d_facexz, n);

  n = Dimensions_size_faceyz( sweeper.dims_b, NU, noctant_per_block) * sizeof(P);

  P* faceyz = (P*) malloc ( n );
  P* d_faceyz = NULL;
  hipMalloc((void**) &d_faceyz, n);

  n = dims.na * NU * dims.ne * NOCTANT * dims.ncell_x * dims.ncell_y * sizeof(P);
  P* vslocal = (P*) malloc ( n ); 

  P* d_vslocal = NULL;
  hipMalloc((void**)&d_vslocal, n );

  // measure host and device execution time
  double ktime = 0.0;
  double k_start, k_end;

  double t1 = get_time();

  for(int iteration=0; iteration<niterations; ++iteration )
  {
    // compute the value of next step
    const int nstep = StepScheduler_nstep( &(sweeper.stepscheduler) );
#ifdef DEBUG
    printf("iteration %d next step = %d\n", iteration, nstep);
#endif

    for (int step = 0; step < nstep; ++step) {

      Dimensions dims = sweeper.dims;
      Dimensions dims_b = sweeper.dims_b;
      int dims_b_ncell_x = dims_b.ncell_x;
      int dims_b_ncell_y = dims_b.ncell_y;
      int dims_b_ncell_z = dims_b.ncell_z;
      int dims_ncell_z = dims.ncell_z;
      int dims_b_ne = dims_b.ne;
      int dims_b_na = dims_b.na;
      //int dims_b_nm = dims_b.nm;

#ifdef DEBUG
      int facexy_size = dims_b.ncell_x * dims_b.ncell_y * dims_b.ne * dims_b.na * NU * NOCTANT;
      int facexz_size = dims_b.ncell_x * dims_b.ncell_z * dims_b.ne * dims_b.na * NU * NOCTANT;
      int faceyz_size = dims_b.ncell_y * dims_b.ncell_z * dims_b.ne * dims_b.na * NU * NOCTANT;
#endif

      int v_size = sizeof(P) * dims.ncell_x * dims.ncell_y * dims.ncell_z * dims.ne * dims.nm * NU;
      //int a_from_m_size = sizeof(P) * dims_b.nm * dims_b.na * NOCTANT;
      //int m_from_a_size = sizeof(P) * dims_b.nm * dims_b.na * NOCTANT;
      //int vs_local_size = sizeof(P) * dims_b.na * NU * dims_b.ne * NOCTANT * dims_b.ncell_x * dims_b.ncell_y;

      int v_b_size = dims_b.ncell_x * dims_b.ncell_y * dims_b.ncell_z * dims_b.ne * dims_b.nm * NU;

      StepInfoAll stepinfoall;  /*---But only use noctant_per_block values---*/

      for(int octant_in_block=0; octant_in_block<noctant_per_block; ++octant_in_block )
      {
        stepinfoall.stepinfo[octant_in_block] = StepScheduler_stepinfo(
            &(sweeper.stepscheduler), step, octant_in_block, 
            0, //proc_x, 
            0  //proc_y 
            );
      }

      const int ix_base = 0;
      const int iy_base = 0;

      const int num_wavefronts = dims_b_ncell_z + dims_b_ncell_y + dims_b_ncell_x - 2;

      const int is_first_step = 0 == step;
      const int is_last_step = nstep - 1 == step;

      dim3 xy_grid((dims_b_ncell_x+7)/8, (dims_b_ncell_y+7)/8, (NOCTANT+3)/4);
      dim3 xy_block(8, 8, 4);

      dim3 xz_grid((dims_b_ncell_x+7)/8, (dims_b_ncell_z+7)/8, (NOCTANT+3)/4);
      dim3 xz_block(8, 8, 4);

      dim3 yz_grid((dims_b_ncell_y+7)/8, (dims_b_ncell_z+7)/8, (NOCTANT+3)/4);
      dim3 yz_block(8, 8, 4);

      dim3 wave_grid((NOCTANT+3)/4, (dims_b_ne+63)/64);
      dim3 wave_block(4, 64);

      if (is_first_step) {
        
        k_start = get_time();

        hipMemset(d_vo, 0, v_size);  // must reset output for each iteration

        hipLaunchKernelGGL(init_facexy, xy_grid, xy_block, 0, 0,  
          ix_base, iy_base, dims_b_ne, dims_b_na, dims_b_ncell_x, 
          dims_b_ncell_y, dims_b_ncell_z, dims_ncell_z, d_facexy);

#ifdef DEBUG
      hipMemcpy(facexy, d_facexy, facexy_size * sizeof(P), hipMemcpyDeviceToHost);
      for (int i = 0; i < facexy_size; i++)
        printf("facexy: %d %f\n", i, facexy[i]);
#endif
      }

      hipLaunchKernelGGL(init_facexz, xz_grid, xz_block, 0, 0, 
          ix_base, iy_base, dims_b_ne, dims_b_na, 
          dims_b_ncell_x, dims_b_ncell_y, dims_ncell_z, 
          1, //proc_x_min, 
          1, //proc_x_max, 
          stepinfoall, d_facexz);

#ifdef DEBUG
      hipMemcpy(facexz, d_facexz, facexz_size * sizeof(P), hipMemcpyDeviceToHost);
      for (int i = 0; i < facexz_size; i++)
        printf("facexz: %d %f\n", i, facexz[i]);
#endif

      hipLaunchKernelGGL(init_faceyz, yz_grid, yz_block, 0, 0, 
          ix_base, iy_base, dims_b_ne, dims_b_na, 
          dims_b_ncell_x, dims_b_ncell_y, dims_ncell_z, 
          1, //proc_y_min, 
          1, //proc_y_max, 
          stepinfoall, d_faceyz);

#ifdef DEBUG
      hipMemcpy(faceyz, d_faceyz, faceyz_size * sizeof(P), hipMemcpyDeviceToHost);
      for (int i = 0; i < faceyz_size; i++)
        printf("faceyz: %d %f\n", i, faceyz[i]);
#endif

      hipLaunchKernelGGL(wavefronts, wave_grid, wave_block, 0, 0, 
          num_wavefronts,  
          ix_base, iy_base, 
          v_b_size,
          noctant_per_block,
          dims_b,
          stepinfoall,
          d_facexy, 
          d_facexz, 
          d_faceyz, 
          d_a_from_m,
          d_m_from_a,
          d_vi,
          d_vo,
          d_vslocal);

      if (is_last_step) { 

        hipDeviceSynchronize();
        k_end = get_time();
        ktime += k_end - k_start;

        hipMemcpy(vo, d_vo, v_size, hipMemcpyDeviceToHost);
#ifdef DEBUG
        for (int i = 0; i < v_size/sizeof(P); i++) printf("vo %d %f\n", i, vo[i]);
#endif
      }
    } // step

    P* tmp = vo;
    vo = vi;
    vi = tmp;
  }
  double t2 = get_time();
  double time = t2 - t1;

  // Verification (input and output vectors are equal) 
  P normsq = (P)0;
  P normsqdiff = (P)0;
  for (size_t i = 0; i < Dimensions_size_state( dims, NU ); i++) {
    normsq += vo[i] * vo[i];
    normsqdiff += (vi[i] - vo[i]) * (vi[i] - vo[i]);
  }
  double flops = niterations *
    ( Dimensions_size_state( dims, NU ) * NOCTANT * 2. * dims.na
      + Dimensions_size_state_angles( dims, NU )
      * Quantities_flops_per_solve( dims )
      + Dimensions_size_state( dims, NU ) * NOCTANT * 2. * dims.na );

  double floprate_h = (time <= 0) ?  0 : flops / (time * 1e-6) / 1e9;
  double floprate_d = (ktime <= 0) ?  0 : flops / (ktime * 1e-6) / 1e9;

  printf( "Normsq result: %.8e  diff: %.3e  verify: %s  host time: %.3f (s) kernel time: %.3f (s)\n",
          normsq,
          normsqdiff,
          normsqdiff== (P)0 ? "PASS" : "FAIL",
  if (normsqdiff != (P)0) exit(1);
          time * 1e-6, ktime * 1e-6);

  printf( "GF/s (host): %.3f\nGF/s (device): %.3f\n", floprate_h, floprate_d );

  /*---Deallocations---*/

  Arguments_destroy( &args );

  hipFree(d_vo);
  hipFree(d_vi);
  hipFree(d_a_from_m);
  hipFree(d_m_from_a);
  hipFree(d_facexy);
  hipFree(d_facexz);
  hipFree(d_faceyz);
  hipFree(d_vslocal);

  free(vi);
  free(vo);
  free(m_from_a);
  free(a_from_m);
  free(facexy);
  free(facexz);
  free(faceyz);
  free(vslocal);

  return 0;
} /*---main---*/
