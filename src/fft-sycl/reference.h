#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440f
#endif

inline T2 reference_exp_i(T phi) {
  return (T2){std::cos(phi), std::sin(phi)};
}

inline T2 reference_cmplx_mul(T2 a, T2 b) {
  return (T2){a.x() * b.x() - a.y() * b.y(),
              a.x() * b.y() + a.y() * b.x()};
}

inline T2 reference_cm_fl_mul(T2 a, T b) {
  return (T2){b * a.x(), b * a.y()};
}

inline T2 reference_cmplx_add(T2 a, T2 b) {
  return (T2){a.x() + b.x(), a.y() + b.y()};
}

inline T2 reference_cmplx_sub(T2 a, T2 b) {
  return (T2){a.x() - b.x(), a.y() - b.y()};
}

inline void reference_fft2(T2* a0, T2* a1) {
  T2 c0 = *a0;
  *a0 = reference_cmplx_add(c0, *a1);
  *a1 = reference_cmplx_sub(c0, *a1);
}

inline void reference_fft4(T2* a0, T2* a1, T2* a2, T2* a3) {
  reference_fft2(a0, a2);
  reference_fft2(a1, a3);
  *a3 = reference_cmplx_mul(*a3, (T2){0, -1});
  reference_fft2(a0, a1);
  reference_fft2(a2, a3);
}

inline void reference_fft8(T2* a) {
  reference_fft2(&a[0], &a[4]);
  reference_fft2(&a[1], &a[5]);
  reference_fft2(&a[2], &a[6]);
  reference_fft2(&a[3], &a[7]);

  a[5] = reference_cm_fl_mul(reference_cmplx_mul(a[5], (T2){1, -1}),
                             M_SQRT1_2);
  a[6] = reference_cmplx_mul(a[6], (T2){0, -1});
  a[7] = reference_cm_fl_mul(reference_cmplx_mul(a[7], (T2){-1, -1}),
                             M_SQRT1_2);

  reference_fft4(&a[0], &a[1], &a[2], &a[3]);
  reference_fft4(&a[4], &a[5], &a[6], &a[7]);
}

template <int SIZE>
void fft1D_512_reference (T2* work, const int n_ffts) 
{
  const int reversed[] = {0,4,2,6,1,5,3,7};
  for (int n = 0; n < n_ffts; n++) { 
    T smem[9*64];
    T2 buffer[8*SIZE];
    T2 *data;
    int i, j, gid, tid, hi, lo;
    for (tid = 0; tid < SIZE; tid++) { 
      gid = n * 512 + tid;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i] = work[gid+i*64];

      reference_fft8( data );

      for(j = 1; j < 8; j++ ){                                       
        data[j] = reference_cmplx_mul( data[j], reference_exp_i(((T)-2*(T)M_PI*reversed[j]/(T)512)*tid) );
      }                                                                   
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].x();
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].x() = smem[lo*66+hi+i*8]; 
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].y();
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].y()= smem[lo*66+hi+i*8]; 
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      data = buffer + tid * 8;
      reference_fft8( data );
      for(j = 1; j < 8; j++ ){                                       
        data[j] = reference_cmplx_mul( data[j], reference_exp_i(((T)-2*(T)M_PI*reversed[j]/(T)64)*hi) );
      }                                                                   
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].x();
    }
    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].x() = smem[hi*72+lo+i*8]; 
    }
    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].y();
    }
    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].y()= smem[hi*72+lo+i*8]; 
    }

    for (tid = 0; tid < SIZE; tid++) {
      data = buffer + tid * 8;
      reference_fft8( data );
      for(i = 0; i < 8; i++ ) {
        gid = n * 512 + tid;
        work[gid+i*64] = data[reversed[i]];
      }
    }
  }
}
