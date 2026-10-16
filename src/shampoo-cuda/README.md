# shampoo

Root-inverse Shampoo on one order-2 parameter block, from
[facebookresearch/optimizers](https://github.com/facebookresearch/optimizers)
`main` at `d24a1491797e5bf2c2985d8789f3a28293d87a6f`,
`distributed_shampoo/preconditioner/shampoo_preconditioner_list.py`.

Each step is the Adam-style Kronecker update from that file
(`beta2 = 0.999`, `weighting_factor = 1 - beta2`, bias correction on, no
dimension normalization), the coupled Newton-Schulz inverse fourth root
(`NewtonSchulzRootInvConfig`, Polar Express schedule, `relative_epsilon = 1e-6`),
and the two-sided precondition `Linv^T G Rinv`. An order-2 block uses inverse
exponent `1/4`, so the root is 4. The amortized inverse root runs on every
timed step.

The direction is then rescaled by AdaGrad grafting from Shi et al.,
arXiv:2309.06497, Section 2.4 and Algorithm 1. The grafted state is the sum
of squared gradient entries (`epsilon = 1e-10`,
`AdaGradPreconditionerConfig` in the same commit). The step multiplies
`Linv^T G Rinv` by `||P_graft||_F / (||P_shampoo||_F + 1e-16)`, the ratio
used by `_precondition_and_grafting`. That copies the grafted step length
onto the Shampoo direction.

This is a different update from `adam` and `adamw` (elementwise moments) and
from `muon` (quintic Newton-Schulz orthogonalization of a rectangular matrix).
The eigenvalue benchmark is tridiagonal bisection, not this dense inverse root.

```text
./main <rows> <cols> <steps> <verify>
```

`make run` checks a rectangular block (`33 31 3 1`) against the OpenMP
reference in `reference.h`, then times one production block (`8192 8192 2 0`)
without that reference. The check uses three steps of a changing gradient,
and `33 x 31` is not a multiple of four, so the scalar tail in the vector
kernels runs. It compares the factor matrices and the preconditioned block
before grafting; grafting alone hides a wrong EMA weight or bias correction.
CMake uses the same check. The timed run keeps one gradient fixed.

CUDA, HIP, and SYCL use the same 256-thread mapping. Contiguous stages load
and store 128 bits per thread, and the Frobenius norm is one 256-wide
reduction over the same groups of four elements. Matrix products are full
fp32 on every port, because this Newton-Schulz iteration diverges if a
product is rounded to TF32 or bf16. CUDA calls `cublasSetMathMode` with
`CUBLAS_PEDANTIC_MATH`. HIP calls `hipblasSetMathMode` with
`HIPBLAS_PEDANTIC_MATH`, and uses `HIPBLAS_DEFAULT_MATH` when rocBLAS
reports that pedantic mode is unsupported. That default is full fp32;
xf32 is a separate mode and is not selected. SYCL calls oneMKL GEMM with
`compute_mode::standard` and sets `MKL_BLAS_COMPUTE_MODE` to the same value
before including oneMKL, so an environment override cannot select TF32.
CUDA, HIP, and SYCL each launch the step directly. The bias correction
`1 - beta2^step` is a host value passed into that launch. SYCL compiles and
links the host OpenMP reference with `-Xarch_host -fopenmp`, so a CUDA or
HIP offload link does not look for a device OpenMP runtime. The CUDA
Makefile targets `sm_70`.

The algorithm is derived from the upstream BSD-licensed sources. See `LICENSE`.
