#include <cfloat>
#include <iostream>
#include <sstream>
#include <chrono>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

using namespace std;


#ifdef SINGLE_PRECISION
#define T float
#define EPISON 1e-4
#else
#define T double
#define EPISON 1e-6
#endif

typedef struct {
  T x;
  T y;
} T2;

#pragma omp declare target

#ifndef M_SQRT1_2
# define M_SQRT1_2      0.70710678118654752440f
#endif


#define exp_1_8   (T2){  1, -1 }//requires post-multiply by 1/sqrt(2)
#define exp_1_4   (T2){  0, -1 }
#define exp_3_8   (T2){ -1, -1 }//requires post-multiply by 1/sqrt(2)

#define iexp_1_8   (T2){  1, 1 }//requires post-multiply by 1/sqrt(2)
#define iexp_1_4   (T2){  0, 1 }
#define iexp_3_8   (T2){ -1, 1 }//requires post-multiply by 1/sqrt(2)

#ifdef SINGLE_PRECISION
inline T2 exp_i( T phi ) {
  return (T2){ cosf(phi), sinf(phi) };
}
#else
inline T2 exp_i( T phi ) {
  return (T2){ cos(phi), sin(phi) };
}
#endif


inline T2 cmplx_mul( T2 a, T2 b ) { return (T2){ a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x }; }
inline T2 cm_fl_mul( T2 a, T  b ) { return (T2){ b*a.x, b*a.y }; }
inline T2 cmplx_add( T2 a, T2 b ) { return (T2){ a.x + b.x, a.y + b.y }; }
inline T2 cmplx_sub( T2 a, T2 b ) { return (T2){ a.x - b.x, a.y - b.y }; }



#define FFT2(a0, a1)                            \
{                                               \
  T2 c0 = *a0;                           \
  *a0 = cmplx_add(c0,*a1);                    \
  *a1 = cmplx_sub(c0,*a1);                    \
}

#define FFT4(a0, a1, a2, a3)                    \
{                                               \
  FFT2( a0, a2 );                             \
  FFT2( a1, a3 );                             \
  *a3 = cmplx_mul(*a3,exp_1_4);               \
  FFT2( a0, a1 );                             \
  FFT2( a2, a3 );                             \
}

#define FFT8(a)                                                 \
{                                                               \
  FFT2( &a[0], &a[4] );                                       \
  FFT2( &a[1], &a[5] );                                       \
  FFT2( &a[2], &a[6] );                                       \
  FFT2( &a[3], &a[7] );                                       \
  \
  a[5] = cm_fl_mul( cmplx_mul(a[5],exp_1_8) , M_SQRT1_2 );    \
  a[6] =  cmplx_mul( a[6] , exp_1_4);                         \
  a[7] = cm_fl_mul( cmplx_mul(a[7],exp_3_8) , M_SQRT1_2 );    \
  \
  FFT4( &a[0], &a[1], &a[2], &a[3] );                         \
  FFT4( &a[4], &a[5], &a[6], &a[7] );                         \
}

#define IFFT2 FFT2

#define IFFT4( a0, a1, a2, a3 )                 \
{                                               \
  IFFT2( a0, a2 );                            \
  IFFT2( a1, a3 );                            \
  *a3 = cmplx_mul(*a3 , iexp_1_4);            \
  IFFT2( a0, a1 );                            \
  IFFT2( a2, a3);                             \
}

#define IFFT8( a )                                              \
{                                                               \
  IFFT2( &a[0], &a[4] );                                      \
  IFFT2( &a[1], &a[5] );                                      \
  IFFT2( &a[2], &a[6] );                                      \
  IFFT2( &a[3], &a[7] );                                      \
  \
  a[5] = cm_fl_mul( cmplx_mul(a[5],iexp_1_8) , M_SQRT1_2 );   \
  a[6] = cmplx_mul( a[6] , iexp_1_4);                         \
  a[7] = cm_fl_mul( cmplx_mul(a[7],iexp_3_8) , M_SQRT1_2 );   \
  \
  IFFT4( &a[0], &a[1], &a[2], &a[3] );                        \
  IFFT4( &a[4], &a[5], &a[6], &a[7] );                        \
}

#pragma omp end declare target

#include "reference.h"

void fft1D_512 (T2* source, const int n_ffts) {
  #pragma omp target teams num_teams(n_ffts) thread_limit(64)
  {
    T smem[8*8*9];
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int blockIdx = omp_get_team_num() * 512 + tid;
      int hi = tid>>3;
      int lo = tid&7;
      T2 data[8];
      const int reversed[] = {0,4,2,6,1,5,3,7};

      // starting index of data to/from global memory
      // globalLoads8(data, source, 64)
      for( int i = 0; i < 8; i++ ) data[i] = source[blockIdx+i*64];

      FFT8( data );

      //twiddle8( data, tid, 512 );
      #ifdef UNROLL
        data[1] = cmplx_mul( data[1],exp_i(((T)-2*(T)M_PI*reversed[1]/(T)512)*tid) );
        data[2] = cmplx_mul( data[2],exp_i(((T)-2*(T)M_PI*reversed[2]/(T)512)*tid) );
        data[3] = cmplx_mul( data[3],exp_i(((T)-2*(T)M_PI*reversed[3]/(T)512)*tid) );
        data[4] = cmplx_mul( data[4],exp_i(((T)-2*(T)M_PI*reversed[4]/(T)512)*tid) );
        data[5] = cmplx_mul( data[5],exp_i(((T)-2*(T)M_PI*reversed[5]/(T)512)*tid) );
        data[6] = cmplx_mul( data[6],exp_i(((T)-2*(T)M_PI*reversed[6]/(T)512)*tid) );
        data[7] = cmplx_mul( data[7],exp_i(((T)-2*(T)M_PI*reversed[7]/(T)512)*tid) );
      #else
        for( int j = 1; j < 8; j++ ){
            data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)512)*tid) );
        }
      #endif

      //transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);
      for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].x;
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) data[i].x = smem[lo*66+hi+i*8];
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].y;
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) data[i].y= smem[lo*66+hi+i*8];
      #pragma omp barrier

      FFT8( data );

      //twiddle8( data, hi, 64 );
      #ifdef UNROLL
        data[1] = cmplx_mul( data[1],exp_i(((T)-2*(T)M_PI*reversed[1]/(T)64)*hi) );
        data[2] = cmplx_mul( data[2],exp_i(((T)-2*(T)M_PI*reversed[2]/(T)64)*hi) );
        data[3] = cmplx_mul( data[3],exp_i(((T)-2*(T)M_PI*reversed[3]/(T)64)*hi) );
        data[4] = cmplx_mul( data[4],exp_i(((T)-2*(T)M_PI*reversed[4]/(T)64)*hi) );
        data[5] = cmplx_mul( data[5],exp_i(((T)-2*(T)M_PI*reversed[5]/(T)64)*hi) );
        data[6] = cmplx_mul( data[6],exp_i(((T)-2*(T)M_PI*reversed[6]/(T)64)*hi) );
        data[7] = cmplx_mul( data[7],exp_i(((T)-2*(T)M_PI*reversed[7]/(T)64)*hi) );
      #else
        for( int j = 1; j < 8; j++ ){
            data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)64)*hi) );
        }
      #endif

      //transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);
      for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].x;
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) data[i].x = smem[hi*72+lo+i*8];
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].y;
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) data[i].y= smem[hi*72+lo+i*8];

      FFT8( data );

      //globalStores8(data, source, 64);
      for( int i = 0; i < 8; i++ )
        source[blockIdx+i*64] = data[reversed[i]];
    }
  }
}

void ifft1D_512 (T2* source, const int n_ffts) {
  #pragma omp target teams num_teams(n_ffts) thread_limit(64)
  {
    T smem[8*8*9];
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int blockIdx = omp_get_team_num() * 512 + tid;
      int hi = tid>>3;
      int lo = tid&7;
      T2 data[8];
      const int reversed[] = {0,4,2,6,1,5,3,7};

      // starting index of data to/from global memory
      for( int i = 0; i < 8; i++ ) data[i] = source[blockIdx+i*64];

      IFFT8( data );

      //itwiddle8( data, tid, 512 );
      #ifdef UNROLL
        data[1] = cmplx_mul( data[1],exp_i(((T)2*(T)M_PI*reversed[1]/(T)512)*tid) );
        data[2] = cmplx_mul( data[2],exp_i(((T)2*(T)M_PI*reversed[2]/(T)512)*tid) );
        data[3] = cmplx_mul( data[3],exp_i(((T)2*(T)M_PI*reversed[3]/(T)512)*tid) );
        data[4] = cmplx_mul( data[4],exp_i(((T)2*(T)M_PI*reversed[4]/(T)512)*tid) );
        data[5] = cmplx_mul( data[5],exp_i(((T)2*(T)M_PI*reversed[5]/(T)512)*tid) );
        data[6] = cmplx_mul( data[6],exp_i(((T)2*(T)M_PI*reversed[6]/(T)512)*tid) );
        data[7] = cmplx_mul( data[7],exp_i(((T)2*(T)M_PI*reversed[7]/(T)512)*tid) );
      #else
        for( int j = 1; j < 8; j++ )
            data[j] = cmplx_mul(data[j] , exp_i(((T)2*(T)M_PI*reversed[j]/(T)512)*(tid)) );
      #endif

      //transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);
      for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].x;
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) data[i].x = smem[lo*66+hi+i*8];
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].y;
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) data[i].y= smem[lo*66+hi+i*8];
      #pragma omp barrier

      IFFT8( data );

      //itwiddle8( data, hi, 64 );
      #ifdef UNROLL
        data[1] = cmplx_mul( data[1],exp_i(((T)2*(T)M_PI*reversed[1]/(T)64)*hi) );
        data[2] = cmplx_mul( data[2],exp_i(((T)2*(T)M_PI*reversed[2]/(T)64)*hi) );
        data[3] = cmplx_mul( data[3],exp_i(((T)2*(T)M_PI*reversed[3]/(T)64)*hi) );
        data[4] = cmplx_mul( data[4],exp_i(((T)2*(T)M_PI*reversed[4]/(T)64)*hi) );
        data[5] = cmplx_mul( data[5],exp_i(((T)2*(T)M_PI*reversed[5]/(T)64)*hi) );
        data[6] = cmplx_mul( data[6],exp_i(((T)2*(T)M_PI*reversed[6]/(T)64)*hi) );
        data[7] = cmplx_mul( data[7],exp_i(((T)2*(T)M_PI*reversed[7]/(T)64)*hi) );
      #else
        for( int j = 1; j < 8; j++ )
            data[j] = cmplx_mul(data[j] , exp_i(((T)2*(T)M_PI*reversed[j]/(T)64)*hi) );
      #endif

      //transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);
      for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].x;
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) data[i].x = smem[hi*72+lo+i*8];
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].y;
      #pragma omp barrier
      for( int i = 0; i < 8; i++ ) data[i].y= smem[hi*72+lo+i*8];

      IFFT8( data );

      for(int i=0; i<8; i++) {
        data[i].x = data[i].x/(T)512;
        data[i].y = data[i].y/(T)512;
      }

      //globalStores8(data, source, 64);
      for( int i = 0; i < 8; i++ )
        source[blockIdx+i*64] = data[reversed[i]];
    }
  }
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <problem size> <number of passes>\n", argv[0]);
    printf("Problem size [0-3]: 0=1M, 1=8M, 2=96M, 3=256M\n");
    return 1;
  }

  srand(2);
  int i;

  int select = atoi(argv[1]);
  int passes = atoi(argv[2]);

  // Convert to MB
  int probSizes[4] = { 1, 8, 96, 256 };
  unsigned long bytes = probSizes[select];
  bytes *= 1024 * 1024;

  // now determine how much available memory will be used
  int half_n_ffts = bytes / (512*sizeof(T2)*2);
  const int n_ffts = half_n_ffts * 2;
  const int half_n_cmplx = half_n_ffts * 512;
  unsigned long used_bytes = half_n_cmplx * 2 * sizeof(T2);
  const int N = half_n_cmplx*2;

  fprintf(stdout, "used_bytes=%lu, N=%d\n", used_bytes, N);

  // allocate host memory, in-place FFT/iFFT operations
  T2 *source = (T2*) malloc (used_bytes);

  // Verification
  T2 *reference = (T2*) malloc (used_bytes);

  // init host memory...
  for (i = 0; i < half_n_cmplx; i++) {
    source[i].x = sinf(i / powf(10000, i % 768 / 384));
    source[i].y = cosf(i / powf(10000, i % 768 / 384));
    source[i+half_n_cmplx].x = source[i].x;
    source[i+half_n_cmplx].y= source[i].y;
  }

  memcpy(reference, source, used_bytes);

  #pragma omp target data map (tofrom: source[0:N])
  {
    fft1D_512(source, n_ffts);

    // verify FFT
    fft1D_512_reference<64>(reference, n_ffts);
    #pragma omp target update from (source[0:N])

    bool error = false;
    for (int i = 0; i < N; i++) {
      if (fabs((T)source[i].x - (T)reference[i].x) > EPISON) {
        //std::cout << i << " " << (T)source[i].x << " " << (T)reference[i].x << std::endl;
        error = true;
        break;
      }
      if (fabs((T)source[i].y - (T)reference[i].y) > EPISON) {
        //std::cout << i << " " << (T)source[i].y << " " << (T)reference[i].y << std::endl;
        error = true;
        break;
      }
    }
    std::cout << "FFT " << (error ? "FAIL" : "PASS")  << std::endl;
    if (error) exit(1);

    ifft1D_512(source, n_ffts);

    // verify iFFT
    #pragma omp target update from (source[0:N])
    error = false;
    for (int i = 0; i < N; i++) {
      int j = i % half_n_cmplx;
      if (fabs((T)source[i].x - (T)sinf(j / powf(10000, j%768/384))) > EPISON) {
        error = true;
        break;
      }
      if (fabs((T)source[i].y - (T)cosf(j / powf(10000, j%768/384))) > EPISON) {
        error = true;
        break;
      }
    }
    std::cout << "iFFT " << (error ? "FAIL" : "PASS")  << std::endl;
    if (error) exit(1);

    auto start = std::chrono::steady_clock::now();

    for (int k=0; k<passes; k++) {
      fft1D_512(source, n_ffts);
      ifft1D_512(source, n_ffts);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Average kernel execution time " << (time * 1e-9f) / passes << " (s)\n";
  }

  free(reference);
  free(source);

  return 0;
}
