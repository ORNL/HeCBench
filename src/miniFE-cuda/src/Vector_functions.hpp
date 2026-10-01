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
#include <cuda.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef MINIFE_HAVE_TBB
#include <LockingVector.hpp>
#endif

#include <TypeTraits.hpp>
#include <Vector.hpp>

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
  template <typename VectorType>
    __global__  void waxby_kernel(
        const int n,
        const typename VectorType::ScalarType alpha,
        const typename VectorType::ScalarType *xcoefs,
        const typename VectorType::ScalarType beta,
        const typename VectorType::ScalarType *ycoefs,
        typename VectorType::ScalarType *wcoefs)
    {
      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) wcoefs[idx] = alpha * xcoefs[idx] + beta * ycoefs[idx];
    }

  template <typename VectorType>
    __global__  void yaxby_kernel(
        const int n,
        const typename VectorType::ScalarType alpha,
        const typename VectorType::ScalarType *xcoefs,
        const typename VectorType::ScalarType beta,
        typename VectorType::ScalarType *ycoefs )
    {
      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) ycoefs[idx] = alpha * xcoefs[idx] + beta * ycoefs[idx];
    }

  template <typename VectorType>
    __global__  void wxby_kernel(
        const int n,
        const typename VectorType::ScalarType *xcoefs,
        const typename VectorType::ScalarType beta,
        const typename VectorType::ScalarType *ycoefs,
        typename VectorType::ScalarType *wcoefs)
    {
      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) wcoefs[idx] = xcoefs[idx] + beta * ycoefs[idx];
    }

  template <typename VectorType>
    __global__  void yxby_kernel(
        const int n,
        const typename VectorType::ScalarType *xcoefs,
        const typename VectorType::ScalarType beta,
        typename VectorType::ScalarType *ycoefs)
    {
      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) ycoefs[idx] = xcoefs[idx] + beta * ycoefs[idx];
    }

  template <typename VectorType>
    __global__  void wx_kernel(
        const int n,
        const typename VectorType::ScalarType *xcoefs,
        typename VectorType::ScalarType *wcoefs)
    {

      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) wcoefs[idx] = xcoefs[idx];
    }

  template <typename VectorType>
    __global__  void dyx_kernel(
        const int n,
        const typename VectorType::ScalarType *xcoefs,
        typename VectorType::ScalarType *ycoefs)
    {

      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) ycoefs[idx] += xcoefs[idx];
    }

  template <typename VectorType>
    __global__  void wax_kernel(
        const int n,
        const typename VectorType::ScalarType alpha,
        const typename VectorType::ScalarType *xcoefs,
        typename VectorType::ScalarType *wcoefs)
    {

      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) wcoefs[idx] = alpha * xcoefs[idx];
    }

  template <typename VectorType>
    __global__  void dyax_kernel(
        const int n,
        const typename VectorType::ScalarType alpha,
        const typename VectorType::ScalarType *xcoefs,
        typename VectorType::ScalarType *ycoefs)
    {

      int idx=blockIdx.x*blockDim.x+threadIdx.x;
      if (idx<n) ycoefs[idx] += alpha * xcoefs[idx];
    }
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
      dim3 grids ((n+255)/256);
      dim3 threads (256);

      if(beta == 0.0) {
        if(alpha == 1.0) {
          wx_kernel<VectorType><<<grids, threads>>>(n, d_xcoefs, d_wcoefs);
        } else {
          wax_kernel<VectorType><<<grids, threads>>>(n, alpha, d_xcoefs, d_wcoefs);
        }
      } else {
        if(alpha == 1.0) {
          wxby_kernel<VectorType><<<grids, threads>>>(n, d_xcoefs, beta, d_ycoefs, d_wcoefs);
        } else {
          waxby_kernel<VectorType><<<grids, threads>>>(n, alpha, d_xcoefs, beta, d_ycoefs, d_wcoefs);
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
        MINIFE_SCALAR *d_xcoefs,
        MINIFE_SCALAR *d_ycoefs)
    {

      const MINIFE_LOCAL_ORDINAL n = MINIFE_MIN(x.coefs.size(), y.coefs.size());

      dim3 grids ((n+255)/256);
      dim3 threads (256);

      if(alpha == 1.0 && beta == 1.0) {
        dyx_kernel<VectorType><<<grids, threads>>>(n, d_xcoefs, d_ycoefs);
      } else if (beta == 1.0) {
        dyax_kernel<VectorType><<<grids, threads>>>(n, alpha, d_xcoefs, d_ycoefs);
      } else if (alpha == 1.0) {
        yxby_kernel<VectorType><<<grids, threads>>>(n, d_xcoefs, beta, d_ycoefs);
      } else if (beta == 0.0) {
        wax_kernel<VectorType><<<grids, threads>>>(n, alpha, d_xcoefs, d_ycoefs);  // reuse the "wax" kernel
      } else {
        yaxby_kernel<VectorType><<<grids, threads>>>(n, alpha, d_xcoefs, beta, d_ycoefs);
      }

    }

  template<typename Scalar, int BLOCK_SIZE>
    __global__ void dot_kernel(const MINIFE_LOCAL_ORDINAL n,
        const Scalar* x,
        const Scalar* y,
              Scalar* d)
    {
      Scalar sum=0;
      for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<n;idx+=gridDim.x*blockDim.x) {
        sum+=x[idx]*y[idx];
      }

      //Do a shared memory reduction on the dot product
      __shared__ Scalar red[BLOCK_SIZE];
      red[threadIdx.x]=sum;
#pragma unroll
      for (int n = BLOCK_SIZE / 2; n > 0; n = n/2) {
        __syncthreads();
        if(threadIdx.x<n)  {sum+=red[threadIdx.x+n]; red[threadIdx.x]=sum;}
      }

      //save partial dot products
      if(threadIdx.x==0) d[blockIdx.x]=sum;
    }

  template<typename Scalar, int BLOCK_SIZE>
    __global__ void final_reduce(Scalar *d) {
      Scalar sum = d[threadIdx.x];
      __shared__ Scalar red[BLOCK_SIZE];

      red[threadIdx.x]=sum;
#pragma unroll
      for (int n = BLOCK_SIZE / 2; n > 0; n = n/2) {
        __syncthreads();
        if(threadIdx.x<n)  {sum+=red[threadIdx.x+n]; red[threadIdx.x]=sum;}
      }
      //save final dot product at the front
      if(threadIdx.x==0) d[0]=sum;
    }


  //-----------------------------------------------------------
  //Compute the dot product of two vectors where:
  //
  // x,y - input vectors
  //
  // result - return-value
  //
  template<typename Vector>
    typename TypeTraits<typename Vector::ScalarType>::magnitude_type
    dot(const Vector& x,
        const Vector& y,
        typename Vector::ScalarType *d_xcoefs,
        typename Vector::ScalarType *d_ycoefs,
        typename Vector::ScalarType *d_dot)
    {
      const MINIFE_LOCAL_ORDINAL n = x.coefs.size();
      typedef typename Vector::ScalarType Scalar;
      Scalar result = 0;
      const int BLOCK_SIZE = 256;
      const int MAX_NUM_BLOCKS = 256;
      const int NUM_BLOCKS = std::min(MAX_NUM_BLOCKS, (n+BLOCK_SIZE-1)/BLOCK_SIZE);
      dim3 grids (NUM_BLOCKS);
      dim3 threads (BLOCK_SIZE);
      // Reuse the caller-provided device scratch/result buffer (MAX_NUM_BLOCKS
      // elements) instead of allocating/freeing it on every dot call.
      Scalar* d = d_dot;
      cudaMemset(d, 0, sizeof(Scalar)*MAX_NUM_BLOCKS);
      dot_kernel<Scalar, BLOCK_SIZE><<<grids, threads>>>(n, d_xcoefs, d_ycoefs, d);
      final_reduce<Scalar, MAX_NUM_BLOCKS><<<1, MAX_NUM_BLOCKS>>>(d);
      cudaMemcpy(&result, d, sizeof(Scalar), cudaMemcpyDeviceToHost);

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
    dot_r2(const Vector& x, const typename Vector::ScalarType *d_xcoefs,
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
      typedef typename Vector::ScalarType Scalar;
      Scalar result = 0;
      const int BLOCK_SIZE = 256;
      const int MAX_NUM_BLOCKS = 256;
      int NUM_BLOCKS = std::min(MAX_NUM_BLOCKS,(n+BLOCK_SIZE-1)/BLOCK_SIZE);
      dim3 grids (NUM_BLOCKS);
      dim3 threads (BLOCK_SIZE);
      // Reuse the caller-provided device scratch/result buffer (MAX_NUM_BLOCKS
      // elements) instead of allocating/freeing it on every dot call.
      Scalar* d = d_dot;
      cudaMemset(d, 0, sizeof(Scalar)*MAX_NUM_BLOCKS);
      dot_kernel<Scalar, BLOCK_SIZE><<<grids, threads>>>(n, d_xcoefs, d_xcoefs, d);
      final_reduce<Scalar, MAX_NUM_BLOCKS><<<1, MAX_NUM_BLOCKS>>>(d);
      cudaMemcpy(&result, d, sizeof(Scalar), cudaMemcpyDeviceToHost);

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

