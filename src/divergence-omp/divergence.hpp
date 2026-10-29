
#ifndef _DIVERGENCE_HPP_
#define _DIVERGENCE_HPP_

#include "../divergence-cuda/divergence_pragmas.hpp"

#define BLOCK_SIZE 16

constexpr const int dim = 2;

template <int np, typename real>
struct element {
  real metdet[np*np];
  real Dinv[np*np*2*2];
  real rmetdet[np*np];
};

template <int np, typename real>
struct derivative {
  real Dvv[np*np];
};

using real = double;


template <int np, typename real>
__attribute__((noinline)) void divergence_sphere_gpu(
    const real *v,
    const derivative<np, real> &deriv,
    const element<np, real> &elem,
          real *div) {

  real gv[np*np*dim]; 

  /* Convert to contra variant form and multiply by g */
  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int k = 0; k < dim; k++) {
        gv[j*np*dim+i*dim+k] = elem.metdet[j*np+i] *
                      (elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim] * v[j*dim*np+dim*i] +
                       elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim+1] * v[j*dim*np+dim*i+1]);
      }
    }
  }

  const real *Dvv = deriv.Dvv;
  const real *rmetdet = elem.rmetdet;
  real vvtemp[np*np];

  #pragma omp target data map(to: gv[0:np*np*dim], \
		Dvv[0:np*np], \
		rmetdet[0:np*np]) \
		map(from: div[0:np*np]) \
		map(alloc: vvtemp[0:np*np])
  {

  #pragma omp target teams distribute parallel for collapse(2) thread_limit(BLOCK_SIZE*BLOCK_SIZE)
  for (int j = 0; j < np; j++) {
    for (int l = 0; l < np; l++) {
        real dudx00 = 0.0;
        real dvdy00 = 0.0;
        for(int i = 0; i < np; i++) {
          dudx00 += Dvv[l*np+i] * gv[j*np*dim+i*dim];
          dvdy00 += Dvv[l*np+i] * gv[i*np*dim+j*dim+1];
        }
        div[j*np+l] = dudx00;
        vvtemp[l*np+j] = dvdy00;
      }
  }

  #pragma omp target teams distribute parallel for collapse(2) thread_limit(BLOCK_SIZE*BLOCK_SIZE)
  for (int l = 0; l < np; l++) {
    for (int j = 0; j < np; j++) {
        div[l*np+j] = (div[l*np+j] + vvtemp[l*np+j]) * 
                      rmetdet[l*np+j] * 1.5683814303638645E-7;
    }
  }

  }
}

template <int np, typename real>
__attribute__((noinline)) void divergence_sphere_cpu(
    const  real *v,
    const derivative<np, real> & deriv,
    const element<np, real> & elem,
     real *div) {
  /* Computes the spherical divergence of v based on the
   * provided metric terms in elem and deriv
   * Returns the divergence in div
   */
  /* Convert to contra variant form and multiply by g */
  real gv[np*np*dim];

  for(int j = 0; j < np; j++) {
    for(int i = 0; i < np; i++) {
      for(int k = 0; k < dim; k++) {
        gv[j*np*dim+i*dim+k] = elem.metdet[j*np+i] *
                      (elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim] * v[j*dim*np+dim*i] +
                       elem.Dinv[j*np*dim*dim+i*dim*dim+k*dim+1] * v[j*dim*np+dim*i+1]);
      }
    }
  }
  /* Compute d/dx and d/dy */
  real vvtemp[np*np];
  for(int l = 0; l < np; l++) {
    for(int j = 0; j < np; j++) {
      real dudx00 = 0;
      real dvdy00 = 0;
      for(int i = 0; i < np; i++) {
        dudx00 += deriv.Dvv[l*np+i] * gv[j*np*dim+i*dim];
        dvdy00 += deriv.Dvv[l*np+i] * gv[i*np*dim+j*dim+1];
      }
      div[j*np+l] = dudx00;
      vvtemp[l*np+j] = dvdy00;
    }
  }
  constexpr const real rrearth = 1.5683814303638645E-7;

  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      div[i*np+j] = (div[i*np+j] + vvtemp[i*np+j]) *
                  (elem.rmetdet[i*np+j] * rrearth);
    }
  }
}

#endif
