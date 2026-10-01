#ifndef SUNDIALS_H
#define SUNDIALS_H

#include <cstddef>

/*
 * SUNDIALS device-kernel miniapp.
 *
 * A single coupled workflow that time-steps a batch of small ODE systems
 * through a representative SUNDIALS implicit integration step, exercising the
 * four GPU hotspots of an implicit integrator on the device. Each kernel is
 * ported directly to CUDA (no cuSPARSE / thrust / cub / MAGMA):
 *
 *   streaming vector ops  CVODE fused N_Vector kernels
 *                         (src/cvode/cvode_fused_gpu.cpp): error-weight setup
 *                         (ewtSetSS/SV), constraint check, nonlinear residual,
 *                         and the CVDiag preconditioner setup/solve steps.
 *
 *   reductions            NVECTOR_CUDA reductions
 *                         (src/nvector/cuda/VectorKernels.cuh): dot product,
 *                         weighted L2 norm squared (WRMS), max norm, L1 norm,
 *                         and min. Each returns a scalar to the host, forcing a
 *                         device/host synchronization for error control.
 *
 *   matvec                Batched block-CSR sparse matrix-vector product
 *                         (src/sunmatrix/cusparse/cusparse_kernels.cuh,
 *                         matvecBCSR): applies the Newton matrix to a vector.
 *
 *   scaleAddI             Batched block-CSR SUNMatScaleAddI (same source):
 *                         A <- c*A + I, forming the Newton matrix M = I -
 *                         gamma*J from the Jacobian each step.
 *
 * The batch holds `nblocks` systems, each of size `m`; the state vector length
 * is n = m * nblocks. One step runs: error-weight setup -> CVDiag setup ->
 * form M (scaleAddI) -> `newton` Newton iterations (matvec + residual) ->
 * WRMS-norm error control (reductions) -> state advance (renormalized to stay
 * bounded, power-iteration style). The GPU run is verified against an identical
 * host reference sequence.
 */

// sunrealtype / sunindextype in SUNDIALS. `real` and `REAL` are aliases.
typedef double REAL;
typedef double real;
typedef int    IDXT;

/*
 * Snapshot of every kernel's output at the final step, used for host/device
 * verification. All array pointers are caller-allocated host buffers; the
 * length-n arrays hold n = m*nblocks elements and `Mbcsr` holds nnz_total =
 * nblocks*blocknnz elements.
 */
typedef struct {
  REAL *ewt;    // ewtSetSS
  REAL *ewtsv;  // ewtSetSV
  REAL *tempv;  // checkConstraints
  REAL *ftemp;  // diagSetupFormY
  REAL *Mdiag;  // diagSetupBuildM
  REAL *Mupd;   // diagSolveUpdateM
  REAL *res;    // nlsResid
  REAL *Mbcsr;  // scaleAddI  (length nnz_total)
  REAL *Ax;     // matvec
  REAL *ycur;   // advanced state
  double dot;   // dotProd(res, ewt)
  double wl2;   // wL2NormSquare(res, ewt)
  double mx;    // maxNorm(res)
  double l1;    // L1Norm(res)
  double mn;    // findMin(res)
} sun_out;

/*
 * All inputs to the miniapp. All arrays are host buffers of length n except the
 * sparsity pattern (rowptr: m+1, colind: blocknnz) and the Jacobian values
 * Jvals (length nnz_total).
 */
typedef struct {
  IDXT m, nblocks, nnz_per_row, blocknnz;
  const IDXT *rowptr, *colind;
  const REAL *Jvals;   // nnz_total
  const REAL *ycur0;   // n  initial state
  const REAL *Vabstol; // n
  const REAL *cflag;   // n  constraint flags
  const REAL *mm;      // n  constraint mask
  const REAL *zn1;     // n
  const REAL *ycor;    // n
  const REAL *fpred;   // n
  const REAL *ypred;   // n
  const REAL *Min;     // n
  // Scalar problem constants
  REAL reltol, Sabstol, rl1, ngamma, h, r, fract, uround, gamma, damp;
} sun_in;

#ifdef __cplusplus
extern "C" {
#endif

// Device miniapp: runs `steps` implicit steps (each with `newton` Newton
// iterations) on resident device buffers and writes the final-step results
// into `out`. Returns the total elapsed device time (nanoseconds) for the whole
// step loop, measured with a single synchronization at each end (no per-step
// synchronizations). block_size is the thread-block size for the
// streaming/reduction kernels.
long sundials_miniapp(int steps, int newton, int block_size, const sun_in *in,
                      sun_out *out);

// Host reference: identical sequence in serial double precision.
void reference(int steps, int newton, const sun_in *in, sun_out *out);

/* Shared helpers (utils.cpp) */

// Build a shared m x m sparsity pattern with `nnz_per_row` nonzeros per row
// (diagonal always present). Returns blocknnz = m * nnz_per_row.
IDXT init_pattern(IDXT m, IDXT nnz_per_row, IDXT *rowptr, IDXT *colind);

// Fill an array with pseudo-random values in [0.5, 1.5].
void init_values(REAL *a, size_t n, unsigned seed);

// Fill an array with pseudo-random values in [lo, hi].
void init_data(real *a, size_t n, unsigned seed, real lo, real hi);

// Relative L1 difference between two arrays.
double check(const REAL *a, const REAL *b, size_t n);

// Relative error between two scalars.
double rel_err(real a, real b);

#ifdef __cplusplus
}
#endif

#endif
