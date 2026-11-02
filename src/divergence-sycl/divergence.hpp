
#ifndef _DIVERGENCE_HPP_
#define _DIVERGENCE_HPP_

#include "../divergence-cuda/divergence_pragmas.hpp"

#include <sycl/sycl.hpp>
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
    sycl::queue &q,
    const  real *v,
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

  real *d_gv = sycl::malloc_device<real>(np*np*dim, q); 
  q.memcpy(d_gv, gv, sizeof(real)*np*np*dim);

  real *d_Dvv = sycl::malloc_device<real>(np*np, q);
  q.memcpy(d_Dvv, deriv.Dvv, sizeof(real)*np*np);

  real *d_rmetdet = sycl::malloc_device<real>(np*np, q);
  q.memcpy(d_rmetdet, elem.rmetdet, sizeof(real)*np*np);

  real *d_div = sycl::malloc_device<real>(np*np, q);
  real *d_vvtemp = sycl::malloc_device<real>(np*np, q);

  sycl::range<2> gws ((np+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE, 
                      (np+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE);
  sycl::range<2> lws (BLOCK_SIZE, BLOCK_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class divergence_test>(
      sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      constexpr const real rrearth = 1.5683814303638645E-7;
      const int l = item.get_global_id(1);
      const int j = item.get_global_id(0); 
      if (l < np && j < np) {
        real dudx00 = 0.0;
        real dvdy00 = 0.0;
        for(int i = 0; i < np; i++) {
          dudx00 += d_Dvv[l*np+i] * d_gv[j*np*dim+i*dim];
          dvdy00 += d_Dvv[l*np+i] * d_gv[i*np*dim+j*dim+1];
        }
        d_div[j*np+l] = dudx00;
        d_vvtemp[l*np+j] = dvdy00;
      }
      item.barrier(sycl::access::fence_space::local_space);

      if (l < np && j < np) 
        d_div[l*np+j] = (d_div[l*np+j] + d_vvtemp[l*np+j]) * 
                        (d_rmetdet[l*np+j] * rrearth);
    });
  });

  q.memcpy(div, d_div, sizeof(real)*np*np).wait();

  sycl::free(d_gv, q);
  sycl::free(d_Dvv, q);
  sycl::free(d_div, q);
  sycl::free(d_vvtemp, q);
  sycl::free(d_rmetdet, q);
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
