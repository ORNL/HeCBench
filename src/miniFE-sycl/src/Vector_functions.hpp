#ifndef _Vector_functions_hpp_
#define _Vector_functions_hpp_

//@HEADER
// ************************************************************************
//
// MiniFE: Simple Finite Element Assembly and Solve
// Copyright (2006-2013) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
//
// ************************************************************************
//@HEADER

#include <vector>
#include <sstream>
#include <fstream>
#include <sycl/sycl.hpp>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef MINIFE_HAVE_TBB
#include <LockingVector.hpp>
#endif

#include <TypeTraits.hpp>
#include <Vector.hpp>
#include <sycl_launch.hpp>

#define MINIFE_MIN(X, Y)  ((X) < (Y) ? (X) : (Y))

namespace miniFE {


template<typename VectorType>
void write_vector(const std::string& filename,
                  const VectorType& vec)
{
  int numprocs = 1, myproc = 0;
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  std::ostringstream osstr;
  osstr << filename << "." << numprocs << "." << myproc;
  std::string full_name = osstr.str();
  std::ofstream ofs(full_name.c_str());

  typedef typename VectorType::ScalarType ScalarType;

  const auto& coefs = vec.coefs;
  for(int p=0; p<numprocs; ++p) {
    if (p == myproc) {
      if (p == 0) {
        ofs << vec.local_size << std::endl;
      }

      typename VectorType::GlobalOrdinalType first = vec.startIndex;
      for(size_t i=0; i<vec.local_size; ++i) {
        ofs << first+i << " " << coefs[i] << std::endl;
      }
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
}

template<typename VectorType>
void sum_into_vector(size_t num_indices,
                     const typename VectorType::GlobalOrdinalType* indices,
                     const typename VectorType::ScalarType* coefs,
                     VectorType& vec)
{
  typedef typename VectorType::GlobalOrdinalType GlobalOrdinal;
  typedef typename VectorType::ScalarType Scalar;

  GlobalOrdinal first = vec.startIndex;
  GlobalOrdinal last = first + vec.local_size - 1;

  auto& vec_coefs = vec.coefs;

  for(size_t i=0; i<num_indices; ++i) {
    if (indices[i] < first || indices[i] > last) continue;
    size_t idx = indices[i] - first;

    #pragma omp atomic
    vec_coefs[idx] += coefs[i];
  }
}

#ifdef MINIFE_HAVE_TBB
template<typename VectorType>
void sum_into_vector(size_t num_indices,
                     const typename VectorType::GlobalOrdinalType* indices,
                     const typename VectorType::ScalarType* coefs,
                     LockingVector<VectorType>& vec)
{
  vec.sum_in(num_indices, indices, coefs);
}
#endif

//------------------------------------------------------------
//Compute the update of a vector with the sum of two scaled vectors where:
//
// w = alpha*x + beta*y
//
// x,y - input vectors
//
// alpha,beta - scalars applied to x and y respectively
//
// w - output vector
//
template<typename VectorType>
void
  waxpby(typename VectorType::ScalarType alpha, const VectorType& x,
         typename VectorType::ScalarType beta, const VectorType& y,
         VectorType& w,
         sycl::queue &q,
         typename VectorType::ScalarType *d_xcoefs,
         typename VectorType::ScalarType *d_ycoefs,
         typename VectorType::ScalarType *d_wcoefs)
{
#ifdef MINIFE_DEBUG
  std::cout << "Starting WAXPBY..." << std::endl;
#endif

#ifdef MINIFE_DEBUG
  if (y.local_size < x.local_size || w.local_size < x.local_size) {
    std::cerr << "miniFE::waxpby ERROR, y and w must be at least as long as x." << std::endl;
    return;
  }
#endif

  const int n = x.coefs.size();
  sycl::range<1> gws ((n+255)/256*256);
  sycl::range<1> lws (256);

  if(beta == 0.0) {
    if(alpha == 1.0) {
      launch_nd(q, sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) d_wcoefs[i] = d_xcoefs[i];
      });
    } else {
      launch_nd(q, sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) d_wcoefs[i] = alpha * d_xcoefs[i];
      });
    }
  } else {
    if(alpha == 1.0) {
      launch_nd(q, sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) d_wcoefs[i] = d_xcoefs[i] + beta * d_ycoefs[i];
      });
    } else {
      launch_nd(q, sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < n) d_wcoefs[i] = alpha * d_xcoefs[i] + beta * d_ycoefs[i];
      });
    }
  }

#ifdef MINIFE_DEBUG
  std::cout << "Finished WAXPBY." << std::endl;
#endif
}

template<typename VectorType>
void
daxpby(const MINIFE_SCALAR alpha,
       const VectorType& x,
       const MINIFE_SCALAR beta,
       VectorType& y,
       sycl::queue &q,
       MINIFE_SCALAR *d_xcoefs,
       MINIFE_SCALAR *d_ycoefs)
{

  const MINIFE_LOCAL_ORDINAL n = MINIFE_MIN(x.coefs.size(), y.coefs.size());

  sycl::range<1> gws ((n+255)/256*256);
  sycl::range<1> lws (256);

  if(alpha == 1.0 && beta == 1.0) {
    launch_nd(q, sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int i = item.get_global_id(0);
      if (i < n) d_ycoefs[i] += d_xcoefs[i];
    });
  } else if (beta == 1.0) {
    launch_nd(q, sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int i = item.get_global_id(0);
      if (i < n) d_ycoefs[i] += alpha * d_xcoefs[i];
    });
  } else if (alpha == 1.0) {
    launch_nd(q, sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int i = item.get_global_id(0);
      if (i < n) d_ycoefs[i] = d_xcoefs[i] + beta * d_ycoefs[i];
    });
  } else if (beta == 0.0) {
    launch_nd(q, sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int i = item.get_global_id(0);
      if (i < n) d_ycoefs[i] = alpha * d_xcoefs[i];
    });
  } else {
    launch_nd(q, sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int i = item.get_global_id(0);
      if (i < n) d_ycoefs[i] = alpha * d_xcoefs[i] + beta * d_ycoefs[i];
    });
  }
}

//-----------------------------------------------------------
//Compute the dot product of two vectors where:
//
// x,y - input vectors
//
// result - return-value
//

#ifndef ONEAPI_REDUCTION
template<typename Scalar, int BLOCK_SIZE>
void dot_kernel(sycl::nd_item<1> &item, Scalar *red,
                const MINIFE_LOCAL_ORDINAL n, const Scalar* x, const Scalar* y, Scalar* d)
{
      Scalar sum = 0;
      int lid = item.get_local_id(0);
      for(int idx=item.get_global_id(0);idx<n;
          idx+=item.get_group_range(0) * item.get_local_range(0)) {
        sum+=x[idx] * y[idx];
      }

      //Do a shared memory reduction on the dot product
      red[lid]=sum;
#pragma unroll
      for (int n = BLOCK_SIZE / 2; n > 0; n = n/2) {
        item.barrier(sycl::access::fence_space::local_space);
        if(lid<n)  {sum+=red[lid+n]; red[lid]=sum;}
      }

      //save partial dot products
      if(lid==0) d[item.get_group(0)]=sum;
}

template<typename Scalar, int BLOCK_SIZE>
void final_reduce(sycl::nd_item<1> &item, Scalar *red, Scalar *d) {
      int lid = item.get_local_id(0);
      MINIFE_SCALAR sum = d[lid];
      red[lid]=sum;
#pragma unroll
      for (int n = BLOCK_SIZE / 2; n > 0; n = n/2) {
        item.barrier(sycl::access::fence_space::local_space);
        if(lid<n)  {sum+=red[lid+n]; red[lid]=sum;}
      }
        //save final dot product at the front
      if(lid==0) d[0]=sum;
}

#endif

template<typename Vector>
typename TypeTraits<typename Vector::ScalarType>::magnitude_type
dot(const Vector& x,
    const Vector& y,
    sycl::queue &q,
    typename Vector::ScalarType *d_xcoefs,
    typename Vector::ScalarType *d_ycoefs,
    typename Vector::ScalarType *d_dot)
{
  const MINIFE_LOCAL_ORDINAL n = x.coefs.size();

  //typedef typename Vector::ScalarType Scalar;
  //const Scalar*  xcoefs = &x.coefs[0];
  //const Scalar*  ycoefs = &y.coefs[0];

  MINIFE_SCALAR result = 0;

#ifdef ONEAPI_REDUCTION
  // Reuse the caller-provided persistent device buffer for the result slot.
  MINIFE_SCALAR *d_result = d_dot;
  q.memcpy(d_result, &result, sizeof(MINIFE_SCALAR));

  sycl::range<1> gws((n+255)/256*256);
  sycl::range<1> lws(256);

  // use SYCL Reduction
  q.submit([&] (sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(gws, lws),
      sycl::reduction(d_result, result, std::plus<MINIFE_SCALAR>()),
      [=] (sycl::nd_item<1> item, auto& res) {
      int i = item.get_global_id(0);
      if (i < n) res += d_xcoefs[i] * d_ycoefs[i];
    });
  });

  q.memcpy(&result, d_result, sizeof(MINIFE_SCALAR)).wait();

#else

  // consistent with the reduction in the miniFE-cuda
  const int BLOCK_SIZE = 256;
  const int MAX_NUM_BLOCKS = 256;
  const int NUM_BLOCKS = std::min(MAX_NUM_BLOCKS, (n+BLOCK_SIZE-1)/BLOCK_SIZE);
  sycl::range<1> gws (NUM_BLOCKS * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  // sum-of-product
  // Reuse the caller-provided persistent device scratch/result buffer
  // (MAX_NUM_BLOCKS elements) instead of allocating/freeing it every call.
  MINIFE_SCALAR *d_sop = d_dot;
  q.memset(d_sop, 0, sizeof(MINIFE_SCALAR)*MAX_NUM_BLOCKS);

  launch_nd_local<MINIFE_SCALAR>(q, sycl::nd_range<1>(gws, lws), BLOCK_SIZE,
    [=] (sycl::nd_item<1> item, MINIFE_SCALAR *red) {
      dot_kernel<MINIFE_SCALAR, BLOCK_SIZE>(item, red, n, d_xcoefs, d_ycoefs, d_sop);
    });

  launch_nd_local<MINIFE_SCALAR>(q, sycl::nd_range<1>(lws, lws), BLOCK_SIZE,
    [=] (sycl::nd_item<1> item, MINIFE_SCALAR *red) {
      final_reduce<MINIFE_SCALAR, MAX_NUM_BLOCKS>(item, red, d_sop);
    });

  q.memcpy(&result, d_sop, sizeof(MINIFE_SCALAR)).wait();
#endif

#ifdef HAVE_MPI
  typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;
  magnitude local_dot = result, global_dot = 0;
  MPI_Datatype mpi_dtype = TypeTraits<magnitude>::mpi_type();
  MPI_Allreduce(&local_dot, &global_dot, 1, mpi_dtype, MPI_SUM, MPI_COMM_WORLD);
  return global_dot;
#else
  return result;
#endif
}

template<typename Vector>
typename TypeTraits<typename Vector::ScalarType>::magnitude_type
dot_r2(const Vector& x, sycl::queue &q, const typename Vector::ScalarType *d_xcoefs,
       typename Vector::ScalarType *d_dot)
{
#ifdef HAVE_MPI
#ifdef MINIFE_DEBUG
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  std::cout << "[" << myrank << "] Starting dot..." << std::endl;
#endif
#endif

  const MINIFE_LOCAL_ORDINAL n = x.coefs.size();

  //typedef typename Vector::ScalarType Scalar;

  MINIFE_SCALAR result = 0;

  //for(int i=0; i<n; ++i) {
  // result += xcoefs[i] * xcoefs[i];
  //}

#ifdef ONEAPI_REDUCTION
  sycl::range<1> gws ((n+255)/256*256);
  sycl::range<1> lws (256);

  // Reuse the caller-provided persistent device buffer for the result slot.
  MINIFE_SCALAR *d_result = d_dot;
  q.memcpy(d_result, &result, sizeof(MINIFE_SCALAR));

  // use SYCL Reduction
  q.submit([&] (sycl::handler &h) {
    h.parallel_for<class reduction_kernel>(sycl::nd_range<1>(gws, lws),
      sycl::reduction(d_result, result, std::plus<MINIFE_SCALAR>()),
      [=] (sycl::nd_item<1> item, auto& res) {
      int i = item.get_global_id(0);
      if (i < n) res += d_xcoefs[i] * d_xcoefs[i];
    });
  });
  q.memcpy(&result, d_result, sizeof(MINIFE_SCALAR)).wait();

#else

  // consistent with the reduction in the miniFE-cuda
  const int BLOCK_SIZE = 256;
  const int MAX_NUM_BLOCKS = 256;
  const int NUM_BLOCKS = std::min(MAX_NUM_BLOCKS, (n+BLOCK_SIZE-1)/BLOCK_SIZE);
  sycl::range<1> gws (NUM_BLOCKS * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  // Reuse the caller-provided persistent device scratch/result buffer
  // (MAX_NUM_BLOCKS elements) instead of allocating/freeing it every call.
  MINIFE_SCALAR *d_sop = d_dot;
  q.memset(d_sop, 0, sizeof(MINIFE_SCALAR)*MAX_NUM_BLOCKS);

  launch_nd_local<MINIFE_SCALAR>(q, sycl::nd_range<1>(gws, lws), BLOCK_SIZE,
    [=] (sycl::nd_item<1> item, MINIFE_SCALAR *red) {
      dot_kernel<MINIFE_SCALAR, BLOCK_SIZE>(item, red, n, d_xcoefs, d_xcoefs, d_sop);
    });

  launch_nd_local<MINIFE_SCALAR>(q, sycl::nd_range<1>(lws, lws), BLOCK_SIZE,
    [=] (sycl::nd_item<1> item, MINIFE_SCALAR *red) {
      final_reduce<MINIFE_SCALAR, MAX_NUM_BLOCKS>(item, red, d_sop);
    });

  q.memcpy(&result, d_sop, sizeof(MINIFE_SCALAR)).wait();

#endif

#ifdef HAVE_MPI
  typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;
  magnitude local_dot = result, global_dot = 0;
  MPI_Datatype mpi_dtype = TypeTraits<magnitude>::mpi_type();
  MPI_Allreduce(&local_dot, &global_dot, 1, mpi_dtype, MPI_SUM, MPI_COMM_WORLD);

#ifdef MINIFE_DEBUG
  std::cout << "[" << myrank << "] Completed dot." << std::endl;
#endif

  return global_dot;
#else
#ifdef MINIFE_DEBUG
  std::cout << "[" << 0 << "] Completed dot." << std::endl;
#endif
  return result;
#endif
}

}//namespace miniFE

#endif

