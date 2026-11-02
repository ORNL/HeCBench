// Copyright (c) 2015 Matthias Noack (ma.noack.pr@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "utils.hpp"

long benchmark(
    complex_t *sigma_in,
    complex_t *sigma_out,
    complex_t *hamiltonian,
    size_t size_sigma,
    size_t size_hamiltonian,
    complex_t *sigma_reference,
    complex_t *sigma_reference_transformed ,
    const int dim, 
    const int num,
    const int kernel_id, 
    size_t vec_length, 
    decltype(&transform_matrices_aos_to_aosoa) transformation_sigma,
    bool scale_hamiltonian,
    decltype(&transform_matrix_aos_to_soa) transformation_hamiltonian)
{
  initialise_hamiltonian(hamiltonian, dim);

  if (scale_hamiltonian) 
    transform_matrix_scale_aos(hamiltonian, dim); // pre-scale hamiltonian

  if (transformation_hamiltonian)
    transformation_hamiltonian(hamiltonian, dim);  

  initialise_sigma(sigma_in, sigma_out, dim, num);

  std::memcpy(sigma_reference_transformed, sigma_reference, size_sigma * sizeof(complex_t));

  // transform memory layout if a transformation is specified
  if (transformation_sigma) {
    // transform reference for comparison
    transformation_sigma(sigma_reference_transformed, dim, num, vec_length);

    // transform sigma
    transformation_sigma(sigma_in, dim, num, vec_length);
  }

  // extract the real and imag data 
  real_2_t* ham = allocate_aligned<real_2_t>(size_hamiltonian);
  real_2_t* sin = allocate_aligned<real_2_t>(size_sigma);
  real_2_t* sout = allocate_aligned<real_2_t>(size_sigma);
  
  for (size_t i = 0; i < size_hamiltonian; i++) {
    ham[i].x = hamiltonian[i].real(); 
    ham[i].y = hamiltonian[i].imag(); 
  }

  for (size_t i = 0; i < size_sigma; i++) {
    sin[i].x = sigma_in[i].real(); 
    sin[i].y = sigma_in[i].imag(); 
  }

  for (size_t i = 0; i < size_sigma; i++) {
    sout[i].x = sigma_out[i].real(); 
    sout[i].y = sigma_out[i].imag(); 
  }

  long total_time = 0;

#pragma omp target data map(from: sout[0:size_sigma]) \
                        map(to: sin[0:size_sigma], \
                                ham[0:size_hamiltonian])
{

  // benchmark loop
  for (size_t i = 0; i < NUM_ITERATIONS; ++i) {

    // clear output 
    #pragma omp target update to (sout[0:size_sigma])

    auto start = std::chrono::steady_clock::now();

    // empty kernel
    switch(kernel_id) {
      case 0:  {
        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {}
        break;
      }

      // initial kernel
      case 1: {
        const real_2_t* __restrict sigma_in = sin;
              real_2_t* __restrict sigma_out = sout;
        const real_2_t* __restrict hamiltonian = ham;

        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {
            int sigma_id = gid * dim * dim;
            for (int i = 0; i < dim; ++i) {
              for (int j = 0; j < dim; ++j) {
                real_2_t tmp;
                tmp.x = 0.0;
                tmp.y = 0.0;
                for (int k = 0; k < dim; ++k) {
                  // z=(x,y), w=(u,v)  z*w = (xu-yv, xv+yu)
                  tmp.x += (hamiltonian[i * dim + k].x * sigma_in[sigma_id + k * dim + j].x - 
                            sigma_in[sigma_id + i * dim + k].x * hamiltonian[k * dim + j].x);
                  tmp.x -= (hamiltonian[i * dim + k].y * sigma_in[sigma_id + k * dim + j].y - 
                            sigma_in[sigma_id + i * dim + k].y * hamiltonian[k * dim + j].y);
                  tmp.y += (hamiltonian[i * dim + k].x * sigma_in[sigma_id + k * dim + j].y - 
                            sigma_in[sigma_id + i * dim + k].x * hamiltonian[k * dim + j].y);
                  tmp.y += (hamiltonian[i * dim + k].y * sigma_in[sigma_id + k * dim + j].x -
                            sigma_in[sigma_id + i * dim + k].y * hamiltonian[k * dim + j].x);
                }
                // multiply with -i * dt / hbar
                sigma_out[sigma_id + i * dim + j].x += hdt * tmp.y;
                sigma_out[sigma_id + i * dim + j].y -= hdt * tmp.x;
              }
            }
          }
        break;
      }

      // refactored initial kernel
      case 2: {
        #define sigma_real(i, j) (sigma_id + 2 * ((i) * dim + (j)))
        #define sigma_imag(i, j) (sigma_id + 2 * ((i) * dim + (j)) + 1)
        
        #define ham_real(i, j) (2 * ((i) * dim + (j)))
        #define ham_imag(i, j) (2 * ((i) * dim + (j)) + 1)

        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;

        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {
          int sigma_id = gid * dim * dim * 2;
          for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
              real_t tmp_real = 0.0;
              real_t tmp_imag = 0.0;
              for (int k = 0; k < dim; ++k) {
                tmp_real += hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                tmp_real -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                tmp_real -= hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_real += sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_imag += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_imag -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_imag += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)]; 
                tmp_imag -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
              }
              // multiply with -i dt/hbar
              sigma_out[sigma_real(i, j)] += hdt * tmp_imag;
              sigma_out[sigma_imag(i, j)] -= hdt * tmp_real;
            }
          }
        }

        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // refactored initial kernel with direct store
      case 3: {
        #define sigma_real(i, j) (sigma_id + 2 * ((i) * dim + (j)))
        #define sigma_imag(i, j) (sigma_id + 2 * ((i) * dim + (j)) + 1)
        #define ham_real(i, j) (2 * ((i) * dim + (j)))
        #define ham_imag(i, j) (2 * ((i) * dim + (j)) + 1)

        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;

        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {
          int sigma_id = gid * dim * dim * 2;
          for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
              for (int k = 0; k < dim; ++k) {
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
                sigma_out[sigma_imag(i, j)] -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_imag(i, j)] += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                sigma_out[sigma_imag(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_imag(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
              }
            }
          }
        }

        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with 1D range
      case 4: {
        #define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * dim * dim)
        #define sigma_id (gid % VEC_LENGTH_AUTO)
        
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + (sigma_id))
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))
        
        #define ham_real(i, j) ((i) * dim + (j))
        #define ham_imag(i, j) (dim * dim + (i) * dim + (j))

        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;

        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {
          for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
              real_t tmp_real = 0.0;
              real_t tmp_imag = 0.0;
              for (int k = 0; k < dim; ++k) {
                tmp_imag -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                tmp_imag += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                tmp_imag += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_imag -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_real += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_real -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_real += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                tmp_real -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
              }
              sigma_out[sigma_real(i, j)] += tmp_real;
              sigma_out[sigma_imag(i, j)] += tmp_imag;
            }
          }
        }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with 1D range and compile time constants
      case 5: {
        #define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * DIM * DIM)
        #define sigma_id (gid % VEC_LENGTH_AUTO)
        
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + (sigma_id))
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))
        
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;

        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {

          for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
              real_t tmp_real = 0.0;
              real_t tmp_imag = 0.0;
              for (int k = 0; k < DIM; ++k) {
                tmp_imag -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                tmp_imag += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                tmp_imag += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_imag -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_real += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_real -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_real += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                tmp_real -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
              }
              sigma_out[sigma_real(i, j)] += tmp_real;
              sigma_out[sigma_imag(i, j)] += tmp_imag;
            }
          }
        }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with 1D range, compile time constants, and permuted loops with temporaries
      case 6: {
        #define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * DIM * DIM)
        #define sigma_id (gid % VEC_LENGTH_AUTO)
        
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + (sigma_id))
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))
        
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;

        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {
          // compute commutator: (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
          for (int i = 0; i < DIM; ++i) {
            for (int k = 0; k < DIM; ++k) {
              real_t ham_real_tmp = hamiltonian[ham_real(i, k)];
              real_t ham_imag_tmp = hamiltonian[ham_imag(i, k)];
              real_t sigma_real_tmp = sigma_in[sigma_real(i, k)];
              real_t sigma_imag_tmp = sigma_in[sigma_imag(i, k)];
              for (int j = 0; j < DIM; ++j) {
                #ifdef USE_INITZERO
                real_t tmp_real = 0.0;
                real_t tmp_imag = 0.0;
                #else
                real_t tmp_real = sigma_out[sigma_real(i, j)];
                real_t tmp_imag = sigma_out[sigma_imag(i, j)];
                #endif
                tmp_imag -= ham_real_tmp * sigma_in[sigma_real(k, j)];
                tmp_imag += sigma_real_tmp * hamiltonian[ham_real(k, j)];
                tmp_imag += ham_imag_tmp * sigma_in[sigma_imag(k, j)];
                tmp_imag -= sigma_imag_tmp * hamiltonian[ham_imag(k, j)];
                tmp_real += ham_real_tmp * sigma_in[sigma_imag(k, j)];
                tmp_real -= sigma_real_tmp * hamiltonian[ham_imag(k, j)];
                tmp_real += ham_imag_tmp * sigma_in[sigma_real(k, j)];
                tmp_real -= sigma_imag_tmp * hamiltonian[ham_real(k, j)];
                #ifdef USE_INITZERO
                sigma_out[sigma_real(i, j)] += tmp_real;
                sigma_out[sigma_imag(i, j)] += tmp_imag;
                #else
                sigma_out[sigma_real(i, j)] = tmp_real;
                sigma_out[sigma_imag(i, j)] = tmp_imag;
                #endif
              }
            }
          }
        }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with 1D range and direct store
      case 7: {
        #define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * dim * dim)
        #define sigma_id (gid % VEC_LENGTH_AUTO)
        
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + (sigma_id))
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))
        #define ham_real(i, j) ((i) * dim + (j))
        #define ham_imag(i, j) (dim * dim + (i) * dim + (j))
            
        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;

        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {
          // compute commutator: (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
          for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
              for (int k = 0; k < dim; ++k) {
                sigma_out[sigma_imag(i, j)] -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_imag(i, j)] += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                sigma_out[sigma_imag(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_imag(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
              }
            }
          }
        }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with 1D range, compile time constants, and direct store
      case 8: {
        #define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * DIM * DIM)
        #define sigma_id (gid % VEC_LENGTH_AUTO)
        
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + (sigma_id))
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;

        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {
          // compute commutator: (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
          for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
              for (int k = 0; k < DIM; ++k) {
                sigma_out[sigma_imag(i, j)] -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_imag(i, j)] += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                sigma_out[sigma_imag(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_imag(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
              }
            }
          }
        }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with 1D range, compile time constants, direct store, and permuted loops with temporaries
      case 9: {
        #define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * DIM * DIM)
        #define sigma_id (gid % VEC_LENGTH_AUTO)
        
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + (sigma_id))
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
            
        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;

        #pragma omp target teams distribute parallel for \
           thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int gid = 0; gid < num; gid++) {
          // compute commutator: (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
          for (int i = 0; i < DIM; ++i) {
            for (int k = 0; k < DIM; ++k) {
              real_t ham_real_tmp = hamiltonian[ham_real(i, k)];
              real_t ham_imag_tmp = hamiltonian[ham_imag(i, k)];
              real_t sigma_real_tmp = sigma_in[sigma_real(i, k)];
              real_t sigma_imag_tmp = sigma_in[sigma_imag(i, k)];
              for (int j = 0; j < DIM; ++j) {
                sigma_out[sigma_imag(i, j)] -= ham_real_tmp * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_imag(i, j)] += sigma_real_tmp * hamiltonian[ham_real(k, j)];
                sigma_out[sigma_imag(i, j)] += ham_imag_tmp * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_imag(i, j)] -= sigma_imag_tmp * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += ham_real_tmp * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_real_tmp * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += ham_imag_tmp * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_imag_tmp * hamiltonian[ham_real(k, j)];
              }
            }
          }
        }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with 2D-range
      case 10: {
        #define package_id ((PACKAGES_PER_WG * g + p) * (VEC_LENGTH_AUTO * 2 * dim * dim))
        #define sigma_id (v)
        
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + sigma_id)
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)
        #define ham_real(i, j) ((i) * dim + (j))
        #define ham_imag(i, j) (dim * dim + (i) * dim + (j))

        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
        
        #pragma omp target teams distribute parallel for collapse(3) \
          thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int g = 0; g < num / VEC_LENGTH_AUTO / PACKAGES_PER_WG; g++)
          for (int p = 0; p < PACKAGES_PER_WG; p++) 
            for (int v = 0; v < VEC_LENGTH_AUTO; v++) {

              for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                  real_t tmp_real = 0.0;
                  real_t tmp_imag = 0.0;
                  for (int k = 0; k < dim; ++k) {
                    tmp_imag -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                    tmp_imag += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                    tmp_imag += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                    tmp_imag -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                    tmp_real += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                    tmp_real -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                    tmp_real += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                    tmp_real -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
                  }
                  sigma_out[sigma_real(i, j)] += tmp_real;
                  sigma_out[sigma_imag(i, j)] += tmp_imag;
                }
              }
            }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with 2D-range and compile-time constants
      case 11: {
        #define package_id ((PACKAGES_PER_WG * g + p) * (VEC_LENGTH_AUTO * 2 * DIM * DIM))
        #define sigma_id (v)
        
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + sigma_id)
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)
        
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
            
        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
        
        #pragma omp target teams distribute parallel for collapse(3) \
          thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int g = 0; g < num / VEC_LENGTH_AUTO / PACKAGES_PER_WG; g++)
          for (int p = 0; p < PACKAGES_PER_WG; p++) 
            for (int v = 0; v < VEC_LENGTH_AUTO; v++) {
              for (int i = 0; i < DIM; ++i) {
                for (int j = 0; j < DIM; ++j) {
                  real_t tmp_real = 0.0;
                  real_t tmp_imag = 0.0;
                  for (int k = 0; k < DIM; ++k) {
                    tmp_imag -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                    tmp_imag += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                    tmp_imag += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                    tmp_imag -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                    tmp_real += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                    tmp_real -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                    tmp_real += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                    tmp_real -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
                  }
                  sigma_out[sigma_real(i, j)] += tmp_real;
                  sigma_out[sigma_imag(i, j)] += tmp_imag;
                }
              }
            }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with 2D-range, compile-time constants, and permuted loops with temporaries
      case 12: {
        #define package_id ((PACKAGES_PER_WG * g + p) * (VEC_LENGTH_AUTO * 2 * DIM * DIM))
        #define sigma_id (v)
        
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + sigma_id)
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
            
        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
        
        #pragma omp target teams distribute parallel for collapse(3) \
          thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int g = 0; g < num / VEC_LENGTH_AUTO / PACKAGES_PER_WG; g++)
          for (int p = 0; p < PACKAGES_PER_WG; p++) 
            for (int v = 0; v < VEC_LENGTH_AUTO; v++) {
              for (int i = 0; i < DIM; ++i) {
                for (int k = 0; k < DIM; ++k) {
                  real_t ham_real_tmp = hamiltonian[ham_real(i, k)];
                  real_t ham_imag_tmp = hamiltonian[ham_imag(i, k)];
                  real_t sigma_real_tmp = sigma_in[sigma_real(i, k)];
                  real_t sigma_imag_tmp = sigma_in[sigma_imag(i, k)];
                  for (int j = 0; j < DIM; ++j) {
#ifdef USE_INITZERO
                    real_t tmp_real = 0.0;
                    real_t tmp_imag = 0.0;
#else
                    real_t tmp_real = sigma_out[sigma_real(i, j)];
                    real_t tmp_imag = sigma_out[sigma_imag(i, j)];
#endif
                    tmp_imag -= ham_real_tmp * sigma_in[sigma_real(k, j)];
                    tmp_imag += sigma_real_tmp * hamiltonian[ham_real(k, j)];
                    tmp_imag += ham_imag_tmp * sigma_in[sigma_imag(k, j)];
                    tmp_imag -= sigma_imag_tmp * hamiltonian[ham_imag(k, j)];
                    tmp_real += ham_real_tmp * sigma_in[sigma_imag(k, j)];
                    tmp_real -= sigma_real_tmp * hamiltonian[ham_imag(k, j)];
                    tmp_real += ham_imag_tmp * sigma_in[sigma_real(k, j)];
                    tmp_real -= sigma_imag_tmp * hamiltonian[ham_real(k, j)];
#ifdef USE_INITZERO
                    sigma_out[sigma_real(i, j)] += tmp_real;
                    sigma_out[sigma_imag(i, j)] += tmp_imag;
#else
                    sigma_out[sigma_real(i, j)] = tmp_real;
                    sigma_out[sigma_imag(i, j)] = tmp_imag;
#endif
                  }
                }
              }
            }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with 2D range and direct store
      case 13: {
        #define package_id ((PACKAGES_PER_WG * g + p) * (VEC_LENGTH_AUTO * 2 * dim * dim))
        #define sigma_id (v)
        
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + sigma_id)
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)
        #define ham_real(i, j) ((i) * dim + (j))
        #define ham_imag(i, j) (dim * dim + (i) * dim + (j))
            
        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
        
        #pragma omp target teams distribute parallel for collapse(3) \
          thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int g = 0; g < num / VEC_LENGTH_AUTO / PACKAGES_PER_WG; g++)
          for (int p = 0; p < PACKAGES_PER_WG; p++) 
            for (int v = 0; v < VEC_LENGTH_AUTO; v++) {
              for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
	          for (int k = 0; k < dim; ++k) {
                    sigma_out[sigma_imag(i, j)] -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                    sigma_out[sigma_imag(i, j)] += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                    sigma_out[sigma_imag(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                    sigma_out[sigma_imag(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                    sigma_out[sigma_real(i, j)] += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                    sigma_out[sigma_real(i, j)] -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                    sigma_out[sigma_real(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                    sigma_out[sigma_real(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
	          }
	        }
              }
            }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with 2D range, compile-time constants, and direct store
      case 14: {
        #define package_id ((PACKAGES_PER_WG * g + p) * (VEC_LENGTH_AUTO * 2 * DIM * DIM))
        #define sigma_id (v)
        
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + sigma_id)
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
            
        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
        
        #pragma omp target teams distribute parallel for collapse(3) \
          thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int g = 0; g < num / VEC_LENGTH_AUTO / PACKAGES_PER_WG; g++)
          for (int p = 0; p < PACKAGES_PER_WG; p++) 
            for (int v = 0; v < VEC_LENGTH_AUTO; v++) {
              for (int i = 0; i < DIM; ++i) {
                for (int j = 0; j < DIM; ++j) {
	          for (int k = 0; k < DIM; ++k) {
                    sigma_out[sigma_imag(i, j)] -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                    sigma_out[sigma_imag(i, j)] += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                    sigma_out[sigma_imag(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                    sigma_out[sigma_imag(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                    sigma_out[sigma_real(i, j)] += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                    sigma_out[sigma_real(i, j)] -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                    sigma_out[sigma_real(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                    sigma_out[sigma_real(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
	          }
	        }
              }
            }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // vectorised kernel with compile-time constants, direct store, and permuted loops with temporaries
      case 15: {
        #define package_id ((PACKAGES_PER_WG * g + p) * (VEC_LENGTH_AUTO * 2 * DIM * DIM))
        #define sigma_id (v)
            
        #define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + sigma_id)
        #define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

        const real_t* __restrict sigma_in = (real_t*)sin;
              real_t* __restrict sigma_out = (real_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
            
        #pragma omp target teams distribute parallel for collapse(3) \
          thread_limit(VEC_LENGTH_AUTO * PACKAGES_PER_WG)
        for (int g = 0; g < num / VEC_LENGTH_AUTO / PACKAGES_PER_WG; g++)
          for (int p = 0; p < PACKAGES_PER_WG; p++) 
            for (int v = 0; v < VEC_LENGTH_AUTO; v++) {
              for (int i = 0; i < DIM; ++i) {
                for (int k = 0; k < DIM; ++k) {
                  real_t ham_real_tmp = hamiltonian[ham_real(i, k)];
                  real_t ham_imag_tmp = hamiltonian[ham_imag(i, k)];
                  real_t sigma_real_tmp = sigma_in[sigma_real(i, k)];
                  real_t sigma_imag_tmp = sigma_in[sigma_imag(i, k)];
	          for (int j = 0; j < DIM; ++j) {
                    sigma_out[sigma_imag(i, j)] -= ham_real_tmp * sigma_in[sigma_real(k, j)];
                    sigma_out[sigma_imag(i, j)] += sigma_real_tmp * hamiltonian[ham_real(k, j)];
                    sigma_out[sigma_imag(i, j)] += ham_imag_tmp * sigma_in[sigma_imag(k, j)];
                    sigma_out[sigma_imag(i, j)] -= sigma_imag_tmp * hamiltonian[ham_imag(k, j)];
                    sigma_out[sigma_real(i, j)] += ham_real_tmp * sigma_in[sigma_imag(k, j)];
                    sigma_out[sigma_real(i, j)] -= sigma_real_tmp * hamiltonian[ham_imag(k, j)];
                    sigma_out[sigma_real(i, j)] += ham_imag_tmp * sigma_in[sigma_real(k, j)];
                    sigma_out[sigma_real(i, j)] -= sigma_imag_tmp * hamiltonian[ham_real(k, j)];
	          }
	        }
              }
            }

        #undef package_id
        #undef sigma_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // manually vectorised kernel
      case 16: {
        #define package_id (gid * dim * dim * 2)
        
        #define sigma_real(i, j) (package_id + 2 * (dim * (i) + (j)))
        #define sigma_imag(i, j) (package_id + 2 * (dim * (i) + (j)) + 1)
        
        #define ham_real(i, j) ((i) * dim + (j))
        #define ham_imag(i, j) (dim * dim + (i) * dim + (j))
    
        const real_vec_t* __restrict sigma_in = (real_vec_t*)sin;
              real_vec_t* __restrict sigma_out = (real_vec_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
          
        #pragma omp target teams distribute parallel for thread_limit(VEC_LENGTH)
        for (int gid = 0; gid < num/VEC_LENGTH; gid++) {
          for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
              real_vec_t tmp_real = v(0.0);
              real_vec_t tmp_imag = v(0.0);
              for (int k = 0; k < dim; ++k) {
                tmp_imag -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                tmp_imag += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                tmp_imag += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_imag -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_real += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_real -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_real += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                tmp_real -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
              }
              sigma_out[sigma_real(i, j)] += tmp_real;
              sigma_out[sigma_imag(i, j)] += tmp_imag;
            }
          }
        }

        #undef package_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // manually vectorised kernel with compile-time constants
      case 17: {
        #define package_id (gid * DIM * DIM * 2)
        
        #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
        #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
        
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
    
        const real_vec_t* __restrict sigma_in = (real_vec_t*)sin;
              real_vec_t* __restrict sigma_out = (real_vec_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
          
        #pragma omp target teams distribute parallel for thread_limit(VEC_LENGTH)
        for (int gid = 0; gid < num/VEC_LENGTH; gid++) {
          for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
              real_vec_t tmp_real = v(0.0);
              real_vec_t tmp_imag = v(0.0);
              for (int k = 0; k < DIM; ++k) {
                tmp_imag -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                tmp_imag += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                tmp_imag += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_imag -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_real += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_real -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_real += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                tmp_real -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
              }
              sigma_out[sigma_real(i, j)] += tmp_real;
              sigma_out[sigma_imag(i, j)] += tmp_imag;
            }
          }
        }

        #undef package_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // manually vectorised kernel with compile-time constants and permuted loops with temporaries
      case 18: {
        #define package_id (gid * DIM * DIM * 2)
        
        #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
        #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
        
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
    
        const real_vec_t* __restrict sigma_in = (real_vec_t*)sin;
              real_vec_t* __restrict sigma_out = (real_vec_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
          
        #pragma omp target teams distribute parallel for thread_limit(VEC_LENGTH)
        for (int gid = 0; gid < num/VEC_LENGTH; gid++) {
          for (int i = 0; i < DIM; ++i) {
            for (int k = 0; k < DIM; ++k) {
              const real_vec_t ham_real_tmp = v(hamiltonian[ham_real(i, k)]);
              const real_vec_t ham_imag_tmp = v(hamiltonian[ham_imag(i, k)]);
              const real_vec_t sigma_real_tmp = sigma_in[sigma_real(i, k)];
              const real_vec_t sigma_imag_tmp = sigma_in[sigma_imag(i, k)];
              for (int j = 0; j < DIM; ++j) {
                #ifdef USE_INITZERO
                real_vec_t tmp_real = v(0.0);
                real_vec_t tmp_imag = v(0.0);
                #else
                real_vec_t tmp_real = sigma_out[sigma_real(i, j)];
                real_vec_t tmp_imag = sigma_out[sigma_imag(i, j)];
                #endif
                tmp_imag -= ham_real_tmp * sigma_in[sigma_real(k, j)];
                tmp_imag += sigma_real_tmp * hamiltonian[ham_real(k, j)];
                tmp_imag += ham_imag_tmp * sigma_in[sigma_imag(k, j)];
                tmp_imag -= sigma_imag_tmp * hamiltonian[ham_imag(k, j)];
                tmp_real += ham_real_tmp * sigma_in[sigma_imag(k, j)];
                tmp_real -= sigma_real_tmp * hamiltonian[ham_imag(k, j)];
                tmp_real += ham_imag_tmp * sigma_in[sigma_real(k, j)];
                tmp_real -= sigma_imag_tmp * hamiltonian[ham_real(k, j)];
                #ifdef USE_INITZERO
                sigma_out[sigma_real(i, j)] += tmp_real;
                sigma_out[sigma_imag(i, j)] += tmp_imag;
                #else
                sigma_out[sigma_real(i, j)] = tmp_real;
                sigma_out[sigma_imag(i, j)] = tmp_imag;
                #endif
              }
            }
          }
        }

        #undef package_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // manually vectorised kernel with compile-time constants and prefetch
      case 19: {
        #define package_id (gid * DIM * DIM * 2)
        
        #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
        #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
        
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
            
        const real_vec_t* __restrict sigma_in = (real_vec_t*)sin;
              real_vec_t* __restrict sigma_out = (real_vec_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
          
        #pragma omp target teams distribute parallel for thread_limit(VEC_LENGTH)
        for (int gid = 0; gid < num/VEC_LENGTH; gid++) {
          for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
              real_vec_t tmp_real = v(0.0);
              real_vec_t tmp_imag = v(0.0);
              for (int k = 0; k < DIM; ++k) {
                tmp_imag -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                tmp_imag += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                tmp_imag += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_imag -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_real += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                tmp_real -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                tmp_real += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                tmp_real -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
              }
              sigma_out[sigma_real(i, j)] += tmp_real;
              sigma_out[sigma_imag(i, j)] += tmp_imag;
            }
          }
        }

        #undef package_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // manually vectorised kernel with direct store
      case 20: {
        #define package_id (gid * dim * dim * 2)
        
        #define sigma_real(i, j) (package_id + 2 * (dim * (i) + (j)))
        #define sigma_imag(i, j) (package_id + 2 * (dim * (i) + (j)) + 1)
        
        #define ham_real(i, j) ((i) * dim + (j))
        #define ham_imag(i, j) (dim * dim + (i) * dim + (j))
    
        const real_vec_t* __restrict sigma_in = (real_vec_t*)sin;
              real_vec_t* __restrict sigma_out = (real_vec_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
          
        #pragma omp target teams distribute parallel for thread_limit(VEC_LENGTH)
        for (int gid = 0; gid < num/VEC_LENGTH; gid++) {
          for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
              for (int k = 0; k < dim; ++k) {
                sigma_out[sigma_imag(i, j)] -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_imag(i, j)] += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                sigma_out[sigma_imag(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_imag(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
              }
            }
          }
        }

        #undef package_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // manually vectorised kernel with compile time constants and direct store
      case 21: {
        #define package_id (gid * DIM * DIM * 2)
        
        #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
        #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
        
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
    
        const real_vec_t* __restrict sigma_in = (real_vec_t*)sin;
              real_vec_t* __restrict sigma_out = (real_vec_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
          
        #pragma omp target teams distribute parallel for thread_limit(VEC_LENGTH)
        for (int gid = 0; gid < num/VEC_LENGTH; gid++) {
          for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
              for (int k = 0; k < DIM; ++k) {
                sigma_out[sigma_imag(i, j)] -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_imag(i, j)] += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                sigma_out[sigma_imag(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_imag(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
              }
            }
          }
        }

        #undef package_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // manually vectorised kernel with compile time constants, direct store, and prefetch
      case 22: {
        #define package_id (gid * DIM * DIM * 2)
        
        #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
        #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
        
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
    
        const real_vec_t* __restrict sigma_in = (real_vec_t*)sin;
              real_vec_t* __restrict sigma_out = (real_vec_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
          
        #pragma omp target teams distribute parallel for thread_limit(VEC_LENGTH)
        for (int gid = 0; gid < num/VEC_LENGTH; gid++) {
          for (int i = 0; i < DIM; ++i) {
            // prefetch result memory for the next inner loops 
            //prefetch(&sigma_out[sigma_real(i, j)], 2 * DIM);
            for (int j = 0; j < DIM; ++j) {
              for (int k = 0; k < DIM; ++k)
              {
                sigma_out[sigma_imag(i, j)] -= hamiltonian[ham_real(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_imag(i, j)] += sigma_in[sigma_real(i, k)] * hamiltonian[ham_real(k, j)];
                sigma_out[sigma_imag(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_imag(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_real(i, k)] * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_real(i, k)] * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += hamiltonian[ham_imag(i, k)] * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_in[sigma_imag(i, k)] * hamiltonian[ham_real(k, j)];
              }
            }
          }
        }

        #undef package_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

       // manually vectorised kernel with compile time constants, direct store, and permuted loops with temporaries
      case 23: {
        #define package_id (gid * DIM * DIM * 2)
        
        #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
        #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
        
        #define ham_real(i, j) ((i) * DIM + (j))
        #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
    
        const real_vec_t* __restrict sigma_in = (real_vec_t*)sin;
              real_vec_t* __restrict sigma_out = (real_vec_t*)sout;
        const real_t* __restrict hamiltonian = (real_t*)ham;
          
        #pragma omp target teams distribute parallel for thread_limit(VEC_LENGTH)
        for (int gid = 0; gid < num/VEC_LENGTH; gid++) {
          for (int i = 0; i < DIM; ++i) {
            for (int k = 0; k < DIM; ++k) {
              real_vec_t ham_real_tmp = v(hamiltonian[ham_real(i, k)]);
              real_vec_t ham_imag_tmp = v(hamiltonian[ham_imag(i, k)]);
              real_vec_t sigma_real_tmp = sigma_in[sigma_real(i, k)];
              real_vec_t sigma_imag_tmp = sigma_in[sigma_imag(i, k)];
              for (int j = 0; j < DIM; ++j) {
                sigma_out[sigma_imag(i, j)] -= ham_real_tmp * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_imag(i, j)] += sigma_real_tmp * hamiltonian[ham_real(k, j)];
                sigma_out[sigma_imag(i, j)] += ham_imag_tmp * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_imag(i, j)] -= sigma_imag_tmp * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += ham_real_tmp * sigma_in[sigma_imag(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_real_tmp * hamiltonian[ham_imag(k, j)];
                sigma_out[sigma_real(i, j)] += ham_imag_tmp * sigma_in[sigma_real(k, j)];
                sigma_out[sigma_real(i, j)] -= sigma_imag_tmp * hamiltonian[ham_real(k, j)];
              }
            }
          }
        }

        #undef package_id
        #undef sigma_real
        #undef sigma_imag
        #undef ham_real
        #undef ham_imag
        break;
      }

      // final GPGPU kernel optimised for an Nvidia GPU
      case 24: {
        #define id_2d_to_1d(i,j) ((i) * DIM + (j))
        #define sigma_id(i,j,m) ((m) * DIM * DIM + ((i) * DIM + (j)))
        #define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

        const real_2_t* __restrict sigma_in = sin;
              real_2_t* __restrict sigma_out = sout;
        const real_2_t* __restrict hamiltonian = ham;

        size_t block_dim_x = (dim * dim + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
        size_t block_dim_y = NUM_SUB_GROUPS;

        const int teams = num / (block_dim_y * CHUNK_SIZE);
        const int threads = block_dim_x * block_dim_y;

        #pragma omp target teams num_teams(teams) thread_limit(threads)
        {
          real_t ham_local_real[DIM*DIM];
          real_t ham_local_imag[DIM*DIM];
          real_t sigma_local_real[2][NUM_SUB_GROUPS][DIM*DIM];
          real_t sigma_local_imag[2][NUM_SUB_GROUPS][DIM*DIM];
          #pragma omp parallel 
	  {
            int tid = omp_get_thread_num();
            int lidx = tid % block_dim_x; 
            int lidy = tid / block_dim_x; 
            int gid = omp_get_team_num();

            // Determine matrix index (i,j) this work item is responsible for
            int ij = lidx; // item.get_local_id(1);
            int i = ij / DIM; // Matrix index 'i' to be processed by this work item in any of 'start -> stop' matrices
            int j = ij % DIM; // Matrix index 'j' to be processed by this work item in any of 'start -> stop' matrices

            // Determine working set : Each work item participates in processing CHUNK_SIZE matrices : 'start -> stop'
            int sub_group_id = lidy; //item.get_local_id(0); // Local matrix ID within work group
            //int start = item.get_group(1) * NUM_SUB_GROUPS * CHUNK_SIZE + sub_group_id * CHUNK_SIZE; // Global matrix ID : start
            int start = gid * NUM_SUB_GROUPS * CHUNK_SIZE + sub_group_id * CHUNK_SIZE; // Global matrix ID : start
            int stop = MIN(num, start + CHUNK_SIZE); // Global matrix ID : stop

            // Local variables
            real_2_t snew1_ij, snew2_ij;
            real_2_t s1, s2;

            // Load Hamiltonian into local memory: only the first sub-group participates
            if (ij < (DIM * DIM) && sub_group_id == 0)
            {
              const real_2_t h = hamiltonian[ij];
              ham_local_real[ij] = h.x;
              ham_local_imag[ij] = h.y;
            }

            // Process all CHUNK_SIZE matrices: two matrices are processed at once (therefore increment 2)
            for (int m = start; m < stop; m += 2)
            {
              #pragma omp barrier
              if (ij < (DIM * DIM)) 
              { // Load input sigma matrix into local memory: only threads with valid IDs participate
                s1 = sigma_in[sigma_id(i, j, m)]; // Real and imaginary part of matrix 'm', element (i,j)
                sigma_local_real[0][sub_group_id][ij] = s1.x;
                sigma_local_imag[0][sub_group_id][ij] = s1.y;

                s2 = sigma_in[sigma_id(i, j, m + 1)]; // Real and imaginary part of matrix 'm+1', element (i,j)
                sigma_local_real[1][sub_group_id][ij] = s2.x;
                sigma_local_imag[1][sub_group_id][ij] = s2.y;

                s1 = sigma_out[sigma_id(i, j, m)]; // Prefetch real and imaginary part of output sigma matrix 'm', element (i,j)
                snew1_ij.x = s1.x;
                snew2_ij.x = s1.y;

                s2 = sigma_out[sigma_id(i, j, m + 1)]; // Prefetch real and imaginary part of output sigma matrix 'm+1', element (i,j)
                snew1_ij.y = s2.x;
                snew2_ij.y = s2.y;
              }
              #pragma omp barrier

              if (ij < (DIM * DIM))
              {
                // Compute commutator: [H,sigma] = H * sigma - sigma * H <=> [H,sigma]_ij = \sum_k ( H_ik * sigma_kj - sigma_ik * H_kj )
                for (int k = 0; k < DIM; ++k)
                {
                  const int ik = id_2d_to_1d(i, k);
                  const int kj = id_2d_to_1d(k, j);

                  // Reassemble real_2_t elements from local memory: 'vector processing' gives better performance here
                  s1 = {sigma_local_real[0][sub_group_id][kj], sigma_local_real[1][sub_group_id][kj]};
                  s2 = {sigma_local_imag[0][sub_group_id][kj], sigma_local_imag[1][sub_group_id][kj]};
                  snew1_ij += ham_local_real[ik] * s2;
                  snew1_ij += ham_local_imag[ik] * s1;
                  snew2_ij -= ham_local_real[ik] * s1;
                  snew2_ij += ham_local_imag[ik] * s2;

                  // Reassemble real_2_t elements from local memory: 'vector processing' gives better performance here
                  s1 = {sigma_local_real[0][sub_group_id][ik], sigma_local_real[1][sub_group_id][ik]};
                  s2 = {sigma_local_imag[0][sub_group_id][ik], sigma_local_imag[1][sub_group_id][ik]};
                  snew1_ij -= ham_local_real[kj] * s2;
                  snew1_ij += ham_local_imag[kj] * s1;
                  snew2_ij += ham_local_real[kj] * s1;
                  snew2_ij -= ham_local_imag[kj] * s2;
                }

                // Write output sigma matrices 'm' and 'm+1', element (i,j)
                sigma_out[sigma_id(i, j, m)] = {snew1_ij.x, snew2_ij.x};
                sigma_out[sigma_id(i, j, m + 1)] = {snew1_ij.y, snew2_ij.y};
              }
            }
          }
        }

        #undef id_2d_to_1d
        #undef sigma_id
        #undef MIN
        break;
      }
      default: std::cerr << "ERROR: **** benchmark kernel unavailable **** \n";
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  std::cout << "Total execution time of kernel "
            << look_up(kernel_id)  << " : " << total_time * 1e-9 << " (s)" << std::endl;
}

  real_t deviation = 0;

  if (kernel_id > 0)  {

    for (size_t i = 0; i < size_sigma; i++) {
      sigma_out[i] = {sout[i].x, sout[i].y};
    }

    // measure the differences between the CPU and GPU results 
    deviation = compare_matrices(sigma_out, sigma_reference_transformed, dim, num);

    std::cout << "Deviation of kernel " << look_up(kernel_id) << ": " << deviation << std::endl;
  } else {
    // the deviation of an empty kernel does not make sense
    std::cout << "Deviation of kernel " << look_up(kernel_id) << "N/A";
  }

  std::cout << std::endl << std::endl;

  free(sin);
  free(sout);
  free(ham);

  return total_time;
}

int main(int argc, char* argv[])
{
  // debugging
  print_compile_config(std::cout);

  // constants
  const size_t dim = DIM;
  const size_t num = NUM;

  // allocate host memory
  size_t size_hamiltonian = dim * dim;
  size_t size_sigma = size_hamiltonian * num;
  size_t size_sigma_byte = sizeof(complex_t) * size_sigma;

  complex_t* hamiltonian = allocate_aligned<complex_t>(size_hamiltonian);
  complex_t* sigma_in = allocate_aligned<complex_t>(size_sigma);
  complex_t* sigma_out = allocate_aligned<complex_t>(size_sigma);
  complex_t* sigma_reference = allocate_aligned<complex_t>(size_sigma);
  complex_t* sigma_reference_transformed = allocate_aligned<complex_t>(size_sigma);

  // perform reference computation for correctness analysis
  initialise_hamiltonian(hamiltonian, dim);
  initialise_sigma(sigma_in, sigma_out, dim, num);
  commutator_reference(sigma_in, sigma_out, hamiltonian, dim, num);

  // copy reference results
  std::memcpy(sigma_reference, sigma_out, size_sigma_byte);

  // total kernel time for all benchmarks 
  long ktime = 0;

  // The macro "BENCHMARK(...)" is defined in utils.hpp
  ktime += BENCHMARK(0, VEC_LENGTH, NO_TRANSFORM, NO_SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(1, VEC_LENGTH, NO_TRANSFORM, NO_SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(2, VEC_LENGTH, NO_TRANSFORM, NO_SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(3, VEC_LENGTH, NO_TRANSFORM, SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(4, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(5, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(6, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(7, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(8, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(9, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(10, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(11, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(12, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(13, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(14, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(15, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(16, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(17, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(18, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(19, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(20, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(21, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(22, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(23, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(24, 2, NO_TRANSFORM, SCALE_HAMILT, NO_TRANSFORM);

  printf("Total kernel time for all benchmarks %lf (s)\n", ktime * 1e-9);

  free(hamiltonian);
  free(sigma_in);
  free(sigma_out);
  free(sigma_reference);
  free(sigma_reference_transformed);

  return 0;
}
