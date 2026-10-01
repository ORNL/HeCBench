#include <chrono>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "sundials.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t err_ = (call);                                                  \
    if (err_ != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,            \
              hipGetErrorString(err_));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define IDX (size_t) blockIdx.x *blockDim.x + threadIdx.x

// AMD wavefronts are 32 (RDNA) or 64 (GCN/CDNA) lanes, so the warp size is read
// from the built-in `warpSize` at runtime rather than hard-coded. MAX_WARPS is
// sized for the largest block we launch (block/warpSize <= 1024/32 = 32).
#define MAX_WARPS 32
#define GRID_STRIDE_XLOOP(type, iter, max)                                     \
  for (type iter = blockDim.x * blockIdx.x + threadIdx.x; iter < (max);        \
       iter += blockDim.x * gridDim.x)

/*
 * =============================================================================
 * Streaming vector kernels (ported from SUNDIALS src/cvode/cvode_fused_gpu.cpp)
 * =============================================================================
 */
// Fused setup: all of ewtSetSS, ewtSetSV, checkConstraints, diagSetupFormY,
// diagSetupBuildM and diagSolveUpdateM. Every original op is elementwise on the
// same index, so fusing them into one kernel keeps the intermediates (ewt,
// ftemp, Mdiag) in registers instead of round-tripping through global memory,
// and collapses six launches into one. Every output is still written so the
// individual ops remain verifiable.
__global__ void k_setup(size_t n, real reltol, real Sabstol, real h, real r,
                        real fract, real uround, const real *ycur,
                        const real *Vabstol, const real *cflag, const real *mm,
                        const real *fpred, const real *zn1, const real *ypred,
                        const real *Min, real *ewt, real *ewtsv, real *tempv,
                        real *save, real *ftemp, real *yform, real *Mdiag,
                        real *ybuild, real *Mupd) {
  size_t i = IDX;
  if (i >= n) return;

  // ewtSetSS / ewtSetSV
  real yc  = ycur[i];
  real w   = 1.0 / (reltol * fabs(yc) + Sabstol);
  ewt[i]   = w;
  ewtsv[i] = 1.0 / (reltol * fabs(yc) + Vabstol[i]);

  // checkConstraints (uses ewt)
  real tmp = (fabs(cflag[i]) >= 1.5) ? 1.0 : 0.0;
  tmp      = tmp * cflag[i] / w;
  save[i]  = -0.1 * tmp;
  tempv[i] = (yc - 0.1 * tmp) * mm[i];

  // diagSetupFormY
  real ft  = h * fpred[i] - zn1[i];
  ftemp[i] = ft;
  yform[i] = r * ft + ypred[i];

  // diagSetupBuildM (uses ftemp, ewt)
  real mval    = fract * ft - h * (Min[i] - fpred[i]);
  real yv      = ft * w;
  bool test    = (fabs(yv) >= uround);
  real bit     = test ? 1.0 : 0.0;
  real bitcomp = test ? 0.0 : -1.0;
  yv           = fract * ft * bit - bitcomp;
  real md      = mval / yv * bit - bitcomp;
  Mdiag[i]     = md;
  ybuild[i]    = yv;

  // diagSolveUpdateM (uses Mdiag)
  Mupd[i] = r * (1.0 / md - 1.0) + 1.0;
}

// N_VScale by a device-resident reciprocal: a[i] *= 1 / denom[0]. Reading the
// norm on the device avoids copying it back to the host every step.
__global__ void k_scaleInv(size_t n, const real *denom, real *a) {
  size_t i = IDX;
  if (i < n) a[i] = a[i] * (1.0 / denom[0]);
}

/*
 * =============================================================================
 * Reductions (ported from SUNDIALS src/nvector/cuda/VectorKernels.cuh and
 * src/sundials/sundials_cuda_kernels.cuh)
 * =============================================================================
 */
struct op_plus {
  __device__ __forceinline__ real operator()(real a, real b) const { return a + b; }
  __device__ static real identity() { return 0.0; }
};
struct op_max {
  __device__ __forceinline__ real operator()(real a, real b) const { return a > b ? a : b; }
  __device__ static real identity() { return -DBL_MAX; }
};
struct op_min {
  __device__ __forceinline__ real operator()(real a, real b) const { return a < b ? a : b; }
  __device__ static real identity() { return DBL_MAX; }
};

__forceinline__ __device__ void atomicOp(op_plus, real *addr, real val) {
  atomicAdd(addr, val);
}
__forceinline__ __device__ void atomicOp(op_max, real *addr, real val) {
  if (*addr >= val) return;
  unsigned long long *p = (unsigned long long *)addr;
  unsigned long long old = *p, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(assumed) >= val) break;
    old = atomicCAS(p, assumed, __double_as_longlong(val));
  } while (assumed != old);
}
__forceinline__ __device__ void atomicOp(op_min, real *addr, real val) {
  if (*addr <= val) return;
  unsigned long long *p = (unsigned long long *)addr;
  unsigned long long old = *p, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(assumed) <= val) break;
    old = atomicCAS(p, assumed, __double_as_longlong(val));
  } while (assumed != old);
}

// Wavefront reduction over `warpSize` lanes (32 or 64 on AMD).
template <typename Op> __inline__ __device__ real warpReduceShflDown(real val) {
  Op op;
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val = op(val, __shfl_down(val, offset));
  return val;
}

template <typename Op>
__inline__ __device__ real blockReduceShflDown(real val, real identity) {
  __shared__ real shared[MAX_WARPS];
  int numThreads = blockDim.x;
  int threadId   = threadIdx.x;
  int warpId     = threadId / warpSize;
  int warpLane   = threadId % warpSize;

  val = warpReduceShflDown<Op>(val);
  if (warpLane == 0) shared[warpId] = val;
  __syncthreads();

  val = (threadId < (numThreads + warpSize - 1) / warpSize) ? shared[threadId]
                                                            : identity;
  if (warpId == 0) val = warpReduceShflDown<Op>(val);
  return val;
}

// Fused reduce-and-advance. In one pass over the vectors this:
//   * computes the five diagnostic reductions over res weighted by ewt
//       out[0]=dot(res,ewt) out[1]=sum((res*ewt)^2) out[2]=max|res|
//       out[3]=sum|res|     out[4]=min(res)
//   * advances the state in place (N_VLinearSum): ycur[i] -= damp*res[i]
//   * reduces the max norm of the advanced state for renormalization
//       out[5]=max|ycur_new|
// This folds the former reduceRes, axpy and renorm-maxNorm kernels together, so
// res and ycur are each read once. Accumulators must be pre-set by k_resetRed.
__global__ void reduceAndAdvance(const real *res, const real *ewt, real *ycur,
                                 real damp, real *out, size_t n) {
  real sdot = 0, swl2 = 0, sl1 = 0, smx = -DBL_MAX, smn = DBL_MAX,
       symx = -DBL_MAX;
  GRID_STRIDE_XLOOP(size_t, i, n) {
    real rv = res[i], p = rv * ewt[i], a = fabs(rv);
    sdot += p; swl2 += p * p; sl1 += a;
    smx = fmax(a, smx); smn = fmin(rv, smn);
    real yn = ycur[i] - damp * rv; // state advance
    ycur[i] = yn;
    symx = fmax(fabs(yn), symx); // norm for renormalization
  }
  // blockReduceShflDown reuses one shared buffer, so sync between reductions.
  sdot = blockReduceShflDown<op_plus>(sdot, 0.0);
  if (threadIdx.x == 0) atomicOp(op_plus{}, out + 0, sdot);
  __syncthreads();
  swl2 = blockReduceShflDown<op_plus>(swl2, 0.0);
  if (threadIdx.x == 0) atomicOp(op_plus{}, out + 1, swl2);
  __syncthreads();
  smx = blockReduceShflDown<op_max>(smx, -DBL_MAX);
  if (threadIdx.x == 0) atomicOp(op_max{}, out + 2, smx);
  __syncthreads();
  sl1 = blockReduceShflDown<op_plus>(sl1, 0.0);
  if (threadIdx.x == 0) atomicOp(op_plus{}, out + 3, sl1);
  __syncthreads();
  smn = blockReduceShflDown<op_min>(smn, DBL_MAX);
  if (threadIdx.x == 0) atomicOp(op_min{}, out + 4, smn);
  __syncthreads();
  symx = blockReduceShflDown<op_max>(symx, -DBL_MAX);
  if (threadIdx.x == 0) atomicOp(op_max{}, out + 5, symx);
}

// Reset all six reduction accumulators to their operator identities in one
// launch (slots 0,1,3 sum; 2,5 max; 4 min).
__global__ void k_resetRed(real *r) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  r[0] = 0.0; r[1] = 0.0; r[2] = -DBL_MAX;
  r[3] = 0.0; r[4] = DBL_MAX; r[5] = -DBL_MAX;
}

/*
 * =============================================================================
 * Batched block-CSR kernels
 * (ported from SUNDIALS src/sunmatrix/cusparse/cusparse_kernels.cuh)
 * =============================================================================
 */
// Batched block-CSR matvec (Ax = A*x) fused with the nonlinear residual
// (nlsResid): res = ngamma*Ax + rl1*zn1 + ycor. Each thread owns one output row
// rowg, so it can write both Ax[rowg] and res[rowg] straight from the register
// sum, folding the former nlsResid kernel into the matvec and avoiding a
// round-trip of Ax through global memory.
__global__ void matvecResidBCSR(IDXT m, IDXT nblocks, IDXT blocknnz,
                                const REAL *A, const IDXT *rowptr,
                                const IDXT *colind, const REAL *x, REAL *y,
                                real rl1, real ngamma, const real *zn1,
                                const real *ycor, real *res) {
  for (IDXT block = blockIdx.x; block < nblocks; block += gridDim.x) {
    for (IDXT row = threadIdx.x; row < m; row += blockDim.x) {
      IDXT tmp    = rowptr[row];
      IDXT rownnz = rowptr[row + 1] - tmp;
      IDXT idxl   = tmp;
      IDXT idxg   = block * blocknnz + tmp;
      IDXT rowg   = block * m + row;
      IDXT colg   = block * m;
      REAL sum    = 0;
      for (IDXT j = 0; j < rownnz; j++)
        sum += A[idxg + j] * x[colg + colind[idxl + j]];
      y[rowg]   = sum;
      res[rowg] = ngamma * sum + (rl1 * zn1[rowg] + ycor[rowg]);
    }
  }
}

// SUNMatScaleAddI, out-of-place: dst = c*src + I on each block. Forming the
// Newton matrix straight from the (constant) Jacobian src into dst avoids a
// separate 30 MB device-to-device copy of src every step.
__global__ void scaleAddIKernelBCSR(IDXT m, IDXT nblocks, IDXT blocknnz, REAL c,
                                    const REAL *src, REAL *dst,
                                    const IDXT *rowptr, const IDXT *colind) {
  for (IDXT block = blockIdx.x; block < nblocks; block += gridDim.x) {
    for (IDXT row = threadIdx.x; row < m; row += blockDim.x) {
      IDXT tmp    = rowptr[row];
      IDXT rownnz = rowptr[row + 1] - tmp;
      IDXT idxl   = tmp;
      IDXT idxg   = block * blocknnz + tmp;
      for (IDXT j = 0; j < rownnz; j++) {
        REAL v = c * src[idxg + j];
        if (colind[idxl + j] == row) v += 1.0;
        dst[idxg + j] = v;
      }
    }
  }
}

/*
 * =============================================================================
 * Host-side driver
 * =============================================================================
 */
#define ALLOC_R(p, count) CHECK(hipMalloc((void **)&(p), (count) * sizeof(real)))
#define ALLOC_I(p, count) CHECK(hipMalloc((void **)&(p), (count) * sizeof(IDXT)))
#define H2D_R(d, h, count)                                                     \
  CHECK(hipMemcpy((d), (h), (count) * sizeof(real), hipMemcpyHostToDevice))
#define H2D_I(d, h, count)                                                     \
  CHECK(hipMemcpy((d), (h), (count) * sizeof(IDXT), hipMemcpyHostToDevice))
#define D2H_R(h, d, count)                                                     \
  CHECK(hipMemcpy((h), (d), (count) * sizeof(real), hipMemcpyDeviceToHost))

// Cap the reduction grid so a modest number of blocks contend on the atomic.
static inline int reduceGrid(size_t n, int bs) {
  size_t g = (n + bs - 1) / bs;
  int cap  = 8 * 1024;
  return (int)(g < (size_t)cap ? g : cap);
}

using clk = std::chrono::steady_clock;
static inline long ns_since(clk::time_point t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0)
      .count();
}

long sundials_miniapp(int steps, int newton, int bs, const sun_in *in,
                      sun_out *out) {
  const IDXT m = in->m, nblocks = in->nblocks, blocknnz = in->blocknnz;
  const size_t n         = (size_t)m * nblocks;
  const size_t nnz_total = (size_t)nblocks * blocknnz;

  // Device buffers (allocated once, resident across all steps).
  IDXT *d_rowptr, *d_colind;
  real *d_ycur, *d_Vabstol, *d_cflag, *d_mm, *d_zn1, *d_ycor, *d_fpred,
      *d_ypred, *d_Min, *d_J;
  real *d_ewt, *d_ewtsv, *d_tempv, *d_save, *d_ftemp, *d_yform, *d_Mdiag,
      *d_ybuild, *d_Mupd, *d_res, *d_Mbcsr, *d_Ax;
  // Reduction accumulators, resident on the device for the whole loop:
  // 0 dot, 1 wL2, 2 maxNorm(res), 3 L1, 4 findMin, 5 maxNorm(ycur, renorm).
  real *d_red;

  ALLOC_I(d_rowptr, m + 1);
  ALLOC_I(d_colind, blocknnz);
  ALLOC_R(d_ycur, n);    ALLOC_R(d_Vabstol, n); ALLOC_R(d_cflag, n);
  ALLOC_R(d_mm, n);      ALLOC_R(d_zn1, n);     ALLOC_R(d_ycor, n);
  ALLOC_R(d_fpred, n);   ALLOC_R(d_ypred, n);   ALLOC_R(d_Min, n);
  ALLOC_R(d_J, nnz_total);
  ALLOC_R(d_ewt, n);     ALLOC_R(d_ewtsv, n);   ALLOC_R(d_tempv, n);
  ALLOC_R(d_save, n);    ALLOC_R(d_ftemp, n);   ALLOC_R(d_yform, n);
  ALLOC_R(d_Mdiag, n);   ALLOC_R(d_ybuild, n);  ALLOC_R(d_Mupd, n);
  ALLOC_R(d_res, n);     ALLOC_R(d_Mbcsr, nnz_total); ALLOC_R(d_Ax, n);
  ALLOC_R(d_red, 6);

  H2D_I(d_rowptr, in->rowptr, m + 1);
  H2D_I(d_colind, in->colind, blocknnz);
  H2D_R(d_ycur, in->ycur0, n);
  H2D_R(d_Vabstol, in->Vabstol, n);
  H2D_R(d_cflag, in->cflag, n);
  H2D_R(d_mm, in->mm, n);
  H2D_R(d_zn1, in->zn1, n);
  H2D_R(d_ycor, in->ycor, n);
  H2D_R(d_fpred, in->fpred, n);
  H2D_R(d_ypred, in->ypred, n);
  H2D_R(d_Min, in->Min, n);
  H2D_R(d_J, in->Jvals, nnz_total);

  dim3 sblock(bs), sgrid((n + bs - 1) / bs);
  dim3 rblock(bs), rgrid(reduceGrid(n, bs));
  int bcsr_bs = 128;
  int bcsr_gr = nblocks < 65535 ? nblocks : 65535;
  dim3 mblock(bcsr_bs), mgrid(bcsr_gr);

  const real reltol = in->reltol, Sabstol = in->Sabstol, rl1 = in->rl1,
             ngamma = in->ngamma, h = in->h, r = in->r, fract = in->fract,
             uround = in->uround, c = -in->gamma, damp = in->damp;

  CHECK(hipDeviceSynchronize());
  clk::time_point start = clk::now();

  for (int s = 0; s < steps; s++) {
    // ---- fused setup: ewtSS/SV + checkConstraints + CVDiag form/build/update ----
    k_setup<<<sgrid, sblock>>>(n, reltol, Sabstol, h, r, fract, uround, d_ycur,
                               d_Vabstol, d_cflag, d_mm, d_fpred, d_zn1,
                               d_ypred, d_Min, d_ewt, d_ewtsv, d_tempv, d_save,
                               d_ftemp, d_yform, d_Mdiag, d_ybuild, d_Mupd);

    // ---- scaleAddI: form Newton matrix M = c*J + I  (BCSR) ----
    scaleAddIKernelBCSR<<<mgrid, mblock>>>(m, nblocks, blocknnz, c, d_J,
                                           d_Mbcsr, d_rowptr, d_colind);

    // ---- Newton iterations: fused matvec (M*ycur) + residual ----
    for (int it = 0; it < newton; it++)
      matvecResidBCSR<<<mgrid, mblock>>>(m, nblocks, blocknnz, d_Mbcsr,
                                         d_rowptr, d_colind, d_ycur, d_Ax, rl1,
                                         ngamma, d_zn1, d_ycor, d_res);

    // ---- fused reduce + advance: 5 res reductions + state advance + renorm
    //      max, all in one pass. Slot 5 holds max|ycur_new|. ----
    k_resetRed<<<1, 1>>>(d_red);
    reduceAndAdvance<<<rgrid, rblock>>>(d_res, d_ewt, d_ycur, damp, d_red, n);
    k_scaleInv<<<sgrid, sblock>>>(n, d_red + 5, d_ycur);
  }

  CHECK(hipDeviceSynchronize());
  long ns_total = ns_since(start);

  // Copy the final-step reduction scalars back once, after timing.
  real hred[6];
  CHECK(hipMemcpy(hred, d_red, 6 * sizeof(real), hipMemcpyDeviceToHost));
  out->dot = hred[0]; out->wl2 = hred[1]; out->mx = hred[2];
  out->l1 = hred[3];  out->mn = hred[4];

  // Copy the final-step results back for verification.
  D2H_R(out->ewt, d_ewt, n);
  D2H_R(out->ewtsv, d_ewtsv, n);
  D2H_R(out->tempv, d_tempv, n);
  D2H_R(out->ftemp, d_ftemp, n);
  D2H_R(out->Mdiag, d_Mdiag, n);
  D2H_R(out->Mupd, d_Mupd, n);
  D2H_R(out->res, d_res, n);
  D2H_R(out->Mbcsr, d_Mbcsr, nnz_total);
  D2H_R(out->Ax, d_Ax, n);
  D2H_R(out->ycur, d_ycur, n);

  CHECK(hipFree(d_rowptr)); CHECK(hipFree(d_colind));
  CHECK(hipFree(d_ycur)); CHECK(hipFree(d_Vabstol)); CHECK(hipFree(d_cflag)); CHECK(hipFree(d_mm));
  CHECK(hipFree(d_zn1)); CHECK(hipFree(d_ycor)); CHECK(hipFree(d_fpred)); CHECK(hipFree(d_ypred));
  CHECK(hipFree(d_Min)); CHECK(hipFree(d_J));
  CHECK(hipFree(d_ewt)); CHECK(hipFree(d_ewtsv)); CHECK(hipFree(d_tempv)); CHECK(hipFree(d_save));
  CHECK(hipFree(d_ftemp)); CHECK(hipFree(d_yform)); CHECK(hipFree(d_Mdiag)); CHECK(hipFree(d_ybuild));
  CHECK(hipFree(d_Mupd)); CHECK(hipFree(d_res)); CHECK(hipFree(d_Mbcsr)); CHECK(hipFree(d_Ax));
  CHECK(hipFree(d_red));
  return ns_total;
}
