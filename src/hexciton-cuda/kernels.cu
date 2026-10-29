__global__ 
void comm_empty(
    real_2_t *__restrict__ sigma_in,
    real_2_t *__restrict__ sigma_out, 
    real_2_t *__restrict__ hamiltonian)
{ 
}


__global__ 
void comm_init (
    const real_2_t *__restrict__ sigma_in,
          real_2_t *__restrict__ sigma_out, 
    const real_2_t *__restrict__ hamiltonian,
    const int dim)
{

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int sigma_id = gid * dim * dim;
  // compute commutator: -i * dt/hbar * (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
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

__global__
void comm_refactor(
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
#define sigma_real(i, j) (sigma_id + 2 * ((i) * dim + (j)))
#define sigma_imag(i, j) (sigma_id + 2 * ((i) * dim + (j)) + 1)

#define ham_real(i, j) (2 * ((i) * dim + (j)))
#define ham_imag(i, j) (2 * ((i) * dim + (j)) + 1)
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
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

#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__
void comm_refactor_direct_store(
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
#define sigma_real(i, j) (sigma_id + 2 * ((i) * dim + (j)))
#define sigma_imag(i, j) (sigma_id + 2 * ((i) * dim + (j)) + 1)
#define ham_real(i, j) (2 * ((i) * dim + (j)))
#define ham_imag(i, j) (2 * ((i) * dim + (j)) + 1)

  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
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

#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa_naive(
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;

#define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * dim * dim)
#define sigma_id (gid % VEC_LENGTH_AUTO)

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + (sigma_id))
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))

#define ham_real(i, j) ((i) * dim + (j))
#define ham_imag(i, j) (dim * dim + (i) * dim + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa_naive_constants (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;

#define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * DIM * DIM)
#define sigma_id (gid % VEC_LENGTH_AUTO)

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + (sigma_id))
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))

#define ham_real(i, j) ((i) * DIM + (j))
#define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa_naive_constants_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
#define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * DIM * DIM)
#define sigma_id (gid % VEC_LENGTH_AUTO)

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + (sigma_id))
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))

#define ham_real(i, j) ((i) * DIM + (j))
#define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa_naive_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;

#define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * dim * dim)
#define sigma_id (gid % VEC_LENGTH_AUTO)

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + (sigma_id))
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))

#define ham_real(i, j) ((i) * dim + (j))
#define ham_imag(i, j) (dim * dim + (i) * dim + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa_naive_constants_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
#define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * DIM * DIM)
#define sigma_id (gid % VEC_LENGTH_AUTO)

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + (sigma_id))
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))

#define ham_real(i, j) ((i) * DIM + (j))
#define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa_naive_constants_direct_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
#define package_id ((gid / VEC_LENGTH_AUTO) * VEC_LENGTH_AUTO * 2 * DIM * DIM)
#define sigma_id (gid % VEC_LENGTH_AUTO)

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + (sigma_id))
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + (sigma_id))

#define ham_real(i, j) ((i) * DIM + (j))
#define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

#define package_id ((PACKAGES_PER_WG * blockIdx.y + threadIdx.y) * (VEC_LENGTH_AUTO * 2 * dim * dim))
#define sigma_id threadIdx.x

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + sigma_id)
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)

#define ham_real(i, j) ((i) * dim + (j))
#define ham_imag(i, j) (dim * dim + (i) * dim + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa_constants (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

#define package_id ((PACKAGES_PER_WG * blockIdx.y + threadIdx.y) * (VEC_LENGTH_AUTO * 2 * DIM * DIM))
#define sigma_id threadIdx.x

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + sigma_id)
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)

#define ham_real(i, j) ((i) * DIM + (j))
#define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa_constants_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

#define package_id ((PACKAGES_PER_WG * blockIdx.y + threadIdx.y) * (VEC_LENGTH_AUTO * 2 * DIM * DIM))
#define sigma_id threadIdx.x

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + sigma_id)
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)

#define ham_real(i, j) ((i) * DIM + (j))
#define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;
#define package_id ((PACKAGES_PER_WG * blockIdx.y + threadIdx.y) * (VEC_LENGTH_AUTO * 2 * dim * dim))
#define sigma_id threadIdx.x

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + sigma_id)
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (dim * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)

#define ham_real(i, j) ((i) * dim + (j))
#define ham_imag(i, j) (dim * dim + (i) * dim + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa_constants_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

#define package_id ((PACKAGES_PER_WG * blockIdx.y + threadIdx.y) * (VEC_LENGTH_AUTO * 2 * DIM * DIM))
#define sigma_id threadIdx.x

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + sigma_id)
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)
#define ham_real(i, j) ((i) * DIM + (j))
#define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_aosoa_constants_direct_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_t *__restrict__ sigma_in = (real_t*) sigma2_in;
  real_t *__restrict__ sigma_out = (real_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

#define package_id ((PACKAGES_PER_WG * blockIdx.y + threadIdx.y) * (VEC_LENGTH_AUTO * 2 * DIM * DIM))
#define sigma_id threadIdx.x

#define sigma_real(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + sigma_id)
#define sigma_imag(i, j) (package_id + 2 * VEC_LENGTH_AUTO * (DIM * (i) + (j)) + VEC_LENGTH_AUTO + sigma_id)

#define ham_real(i, j) ((i) * DIM + (j))
#define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

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

#undef package_id
#undef sigma_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_manual_aosoa (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // number of package to process == get_global_id(0)
#define package_id (gid * dim * dim * 2)

#define sigma_real(i, j) (package_id + 2 * (dim * (i) + (j)))
#define sigma_imag(i, j) (package_id + 2 * (dim * (i) + (j)) + 1)

#define ham_real(i, j) ((i) * dim + (j))
#define ham_imag(i, j) (dim * dim + (i) * dim + (j))

  // compute commutator: (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
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

#undef package_id
#undef sigma_real
#undef sigma_imag
#undef ham_real
#undef ham_imag
}

__global__ 
void comm_manual_aosoa_constants (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  #define package_id (gid * DIM * DIM * 2)
  
  #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
  #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
  
  #define ham_real(i, j) ((i) * DIM + (j))
  #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
  
  // compute commutator: (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
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

  #undef package_id
  #undef sigma_real
  #undef sigma_imag
  #undef ham_real
  #undef ham_imag
}

__global__ 
void comm_manual_aosoa_constants_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  #define package_id (gid * DIM * DIM * 2)
  
  #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
  #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
  
  #define ham_real(i, j) ((i) * DIM + (j))
  #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
  
  // compute commutator: (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
  for (int i = 0; i < DIM; ++i) {
    for (int k = 0; k < DIM; ++k) {
      real_vec_t ham_real_tmp = v(hamiltonian[ham_real(i, k)]);
      real_vec_t ham_imag_tmp = v(hamiltonian[ham_imag(i, k)]);
      real_vec_t sigma_real_tmp = sigma_in[sigma_real(i, k)];
      real_vec_t sigma_imag_tmp = sigma_in[sigma_imag(i, k)];
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

  #undef package_id
  #undef sigma_real
  #undef sigma_imag
  #undef ham_real
  #undef ham_imag
}

__global__ 
void comm_manual_aosoa_constants_perm_prefetch (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  #define package_id (gid * DIM * DIM * 2)
  
  #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
  #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
  
  #define ham_real(i, j) ((i) * DIM + (j))
  #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))

  // compute commutator: (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
  for (int i = 0; i < DIM; ++i) {
    int j = 0;
    //(sigma_out.get_pointer() + sigma_real(i, j)).prefetch(2 * DIM);
    for (j = 0; j < DIM; ++j) {
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

  #undef package_id
  #undef sigma_real
  #undef sigma_imag
  #undef ham_real
  #undef ham_imag
}

__global__ 
void comm_manual_aosoa_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2,
    const int dim)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  #define package_id (gid * dim * dim * 2)
  
  #define sigma_real(i, j) (package_id + 2 * (dim * (i) + (j)))
  #define sigma_imag(i, j) (package_id + 2 * (dim * (i) + (j)) + 1)
  
  #define ham_real(i, j) ((i) * dim + (j))
  #define ham_imag(i, j) (dim * dim + (i) * dim + (j))
  
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

  #undef package_id
  #undef sigma_real
  #undef sigma_imag
  #undef ham_real
  #undef ham_imag
}

__global__ 
void comm_manual_aosoa_constants_direct (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  #define package_id (gid * DIM * DIM * 2)
  
  #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
  #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
  
  #define ham_real(i, j) ((i) * DIM + (j))
  #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
  
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

  #undef package_id
  #undef sigma_real
  #undef sigma_imag
  #undef ham_real
  #undef ham_imag
}

__global__ 
void comm_manual_aosoa_constants_direct_prefetch (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  #define package_id (gid * DIM * DIM * 2)
  #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
  #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
  
  #define ham_real(i, j) ((i) * DIM + (j))
  #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
  
  // compute commutator: (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
  for (int i = 0; i < DIM; ++i) {
    // prefetch result memory for the next inner loops 
    int j = 0;
    //prefetch(&sigma_out[sigma_real(i, j)], 2 * DIM);
    //(sigma_out.get_pointer() + sigma_real(i, j)).prefetch(2 * DIM);
    for (j = 0; j < DIM; ++j) {
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

  #undef package_id
  #undef sigma_real
  #undef sigma_imag
  #undef ham_real
  #undef ham_imag
}

__global__ 
void comm_manual_aosoa_constants_direct_perm (
    const real_2_t *__restrict__ sigma2_in,
          real_2_t *__restrict__ sigma2_out, 
    const real_2_t *__restrict__ hamiltonian2)
{
  real_vec_t *__restrict__ sigma_in = (real_vec_t*) sigma2_in;
  real_vec_t *__restrict__ sigma_out = (real_vec_t*) sigma2_out;
  real_t *__restrict__ hamiltonian = (real_t*) hamiltonian2;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  #define package_id (gid * DIM * DIM * 2)
  
  #define sigma_real(i, j) (package_id + 2 * (DIM * (i) + (j)))
  #define sigma_imag(i, j) (package_id + 2 * (DIM * (i) + (j)) + 1)
  
  #define ham_real(i, j) ((i) * DIM + (j))
  #define ham_imag(i, j) (DIM * DIM + (i) * DIM + (j))
  
  // compute commutator: (hamiltonian * sigma_in[sigma_id] - sigma_in[sigma_id] * hamiltonian)
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

  #undef package_id
  #undef sigma_real
  #undef sigma_imag
  #undef ham_real
  #undef ham_imag
}

__global__ 
void final_gpu_kernel (
    const real_2_t *__restrict__ sigma_in,
          real_2_t *__restrict__ sigma_out, 
    const real_2_t *__restrict__ hamiltonian,
    const int num)
{
  #define id_2d_to_1d(i,j) ((i) * DIM + (j))
  #define sigma_id(i,j,m) ((m) * DIM * DIM + ((i) * DIM + (j)))
  #define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
  // Local memory: shared between all work items in the same work group
  // 2-way shared memory bank conflicts will occur for real_t = double
  // real parts and imaginary parts are stored separately to avoid 4-way bank conflicts in case of real_2_t = double2
  // Input sigma matrix: real part (2 matrices are processed at once)
  // Input sigma matrix: imag part (2 matrices are processed at once)
  __shared__ real_t ham_local_real[DIM*DIM];
  __shared__ real_t ham_local_imag[DIM*DIM];
  __shared__ real_t sigma_local_real[2][NUM_SUB_GROUPS][DIM*DIM];
  __shared__ real_t sigma_local_imag[2][NUM_SUB_GROUPS][DIM*DIM];

  // Determine matrix index (i,j) this work item is responsible for
  int ij = threadIdx.x;
  int i = ij / DIM; // Matrix index 'i' to be processed by this work item in any of 'start -> stop' matrices
  int j = ij % DIM; // Matrix index 'j' to be processed by this work item in any of 'start -> stop' matrices

  // Determine working set : Each work item participates in processing CHUNK_SIZE matrices : 'start -> stop'
  int sub_group_id = threadIdx.y; // Local matrix ID within work group
  int start = blockIdx.x * NUM_SUB_GROUPS * CHUNK_SIZE + sub_group_id * CHUNK_SIZE; // Global matrix ID : start
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
    __syncthreads();
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
    __syncthreads();

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

  #undef id_2d_to_1d
  #undef sigma_id
  #undef MIN
}
