// Copyright (c) 2015 Matthias Noack (ma.noack.pr@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef UTILS_HPP
#define UTILS_HPP

#include <complex>
#include <iostream>
#include <cstring> // memcpy
#include <cstdlib>
#include <chrono>
#include <hip/hip_runtime.h>

#define SINGLE_PRECISION

#ifdef SINGLE_PRECISION
using real_t = float;
using real_2_t = float2;
// == 1 / Pi
#define hbar  (1.0f / acosf(-1.f)) 
#define dt  1.0e-3f
#define hdt  (dt / hbar)


#else
using real_t = double;
using real_2_t = double2;
#define hbar  (1.0 / acos(-1.0))
#define dt  1.0e-3
#define hdt  (dt / hbar)

#endif

using complex_t = std::complex<real_t>;

#define VEC_LENGTH_AUTO 4

#ifndef VEC_LENGTH 
#define VEC_LENGTH VEC_LENGTH_AUTO  
#endif

#ifdef SINGLE_PRECISION
#if  VEC_LENGTH == 2
  #define real_vec_t float2
  #define v(x) make_float2(x, x)
#elif VEC_LENGTH == 4
  #define real_vec_t float4
  #define v(x) make_float4(x, x, x, x)
#else  // float8 and float16 are undefined in CUDA/HIP
  #define real_vec_t float4
  #define v(x) make_float4(x, x, x, x)
#endif
#else
#if  VEC_LENGTH == 2
  #define real_vec_t double2
  #define v(x) make_double2(x, x)
#elif VEC_LENGTH == 4
  #define real_vec_t double4
  #define v(x) make_double4(x, x, x, x)
#else  // double8 and double16 are undefined in CUDA/HIP
  #define real_vec_t double4
  #define v(x) make_double4(x, x, x, x)
#endif
#endif

#ifndef PACKAGES_PER_WG 
#define PACKAGES_PER_WG 64
#endif

// alignment for memory allocations
#ifndef DEFAULT_ALIGNMENT
#define DEFAULT_ALIGNMENT 64
#endif

// number of kernel iterations (including warmup)
#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 1001
#endif

// number of warmup iterations not taken into statistics
#ifndef NUM_WARMUP
#define NUM_WARMUP 1
#endif  

// matrix dimension (based on actual application value)
#ifndef DIM
#define DIM 7
#endif

// number of matrices in the sigma vectors (based on actual application value)
#ifndef NUM
//#define NUM 512*1024
#define NUM 2048
#endif

#ifndef PACKAGES_PER_WG
#define PACKAGES_PER_WG 4
#endif
#ifndef NUM_SUB_GROUPS
#define NUM_SUB_GROUPS 2
#endif
#ifndef CHUNK_SIZE
#define CHUNK_SIZE 16
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// kernel identifier, vector length, transform and scale
#define BENCHMARK(KID, VLEN, X1, SCALE, X2) \
  benchmark(sigma_in, sigma_out, hamiltonian, size_sigma, size_hamiltonian, \
            sigma_reference, sigma_reference_transformed,\
            dim, num, KID, VLEN, X1, SCALE, X2)
 

constexpr std::nullptr_t NO_TRANSFORM = nullptr;
constexpr bool NO_SCALE_HAMILT = false;
constexpr bool SCALE_HAMILT = true;

void print_compile_config(std::ostream& out);

// aligned_alloc from C++11 is not available for the Phi
  template<typename T>
T* allocate_aligned(size_t size, size_t alignment = DEFAULT_ALIGNMENT)
{
  T* ptr = nullptr;
  int err = 0;
  if ((err = posix_memalign((void**)&ptr, alignment, size * sizeof(T))))
    std::cerr << "Error: posix_memalign() returned: " << err << std::endl;
  return ptr;
}

// reference kernel signature
void commutator_reference(complex_t* sigma_in, complex_t* sigma_out, complex_t* hamiltonian, size_t dim, size_t num_sigma);

// intitialise sigma matrices
void initialise_sigma(complex_t* sigma_in, complex_t* sigma_out, size_t dim, size_t num);

// initialise hamiltonian
void initialise_hamiltonian(complex_t* hamiltonian, size_t dim);

// scale matrix by factor
void transform_matrix_scale_aos(complex_t* matrix, size_t dim);

// transform matrix format of a complex matrix from array of structs (AoS)
// RIRIRI... to struct of array (SoA) RRR...III...
// AoS: 
//     struct complex_t { real x, y; };
//     complex_t matrix[size];
// SoA: 
//     struct { real x[size], y[size]; } matrix;
void transform_matrix_aos_to_soa(complex_t* matrix, size_t dim);

// transform a vector of complex AoS matrices into an interleaved hybrid SoA
// format (AoSoA) with an inner size of the SIMD-width specified by VEC_LENGTH
// RIRIRI...RIRIRI... => RRR...III...RRR...III..
// AoS:
//     struct complex_t { real x, y; }; 
//     complex_t matrices[size * num];
// AoSoA: 
//     struct complex_t { real x[VEC_LENGTH], y[VEC_LENGTH]; }; 
//     complex_t matrix[size * num / VEC_LENGTH];
void transform_matrices_aos_to_aosoa(complex_t* matrices, size_t dim, size_t num, size_t vec_length = VEC_LENGTH);

// similar to transform_matrices_aos_to_aosoa, but packs real and imaginary
// parts of complex numbers differently:
// stores packages of interleaved matrices, with all real parts of the package
// preceding all the imaginare parts
void transform_matrices_aos_to_aosoa_gpu(complex_t* matrices, size_t dim, size_t num, size_t vec_length = VEC_LENGTH);

// returns the sum of the absolute values of the element-wise differences as
// measure of deviation
real_t compare_matrices(complex_t* a, complex_t* b, size_t dim, size_t num);

// look up the name corresponding to the kernel id
std::string look_up(const int kid);


#endif // UTILS_HPP

