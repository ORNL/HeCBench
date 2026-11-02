# laghos-force

This benchmark extracts the 3D partial-assembly force action from the ECP
proxy application [Laghos](https://github.com/CEED/Laghos). It is based on
`ForceMult3D` in `laghos_assembly.cpp` at upstream commit
`d2c766e961541f0e1d826aa30aceeadcc05e88c2`. Laghos identifies the partial-
assembly `MassPAOperator` and `ForcePAOperator` actions as its main
computational kernels.

`--points` selects one of the four 3D sizes specialized by Laghos. Each case
uses the positive Bernstein thermodynamic basis, Gauss–Lobatto kinematic
basis, and Gauss–Legendre quadrature from that Laghos configuration:

| `--points` | Laghos order | `D1D` | `Q1D` | `L1D` | Default elements |
| --- | --- | --- | --- | --- | --- |
| 8 | Q1–Q0 | 2 | 2 | 1 | 524,288 |
| 64 | Q2–Q1, the default and ATS priority 1 | 3 | 4 | 2 | 65,536 |
| 216 | Q3–Q2, ATS priority 2 | 4 | 6 | 3 | 19,418 |
| 512 | Q4–Q3 | 5 | 8 | 4 | 8,192 |

The kernel interpolates internal energy to the selected quadrature grid,
multiplies it by the nine components of `stressJinvT`, and applies the three
tensor-product derivatives. A direct host implementation evaluates the same
operator and checks eight elements after timing. All sizes use double precision
and the Laghos/MFEM tensor layout.

## Optimization

Laghos's generic kernel processes the three output components serially and
uses synchronization between every directional contraction. This
specialization contracts all three components and directions together. It
reuses energy interpolated once per element, keeps the basis and intermediates
in shared/local memory, uses compile-time loop bounds, and needs only four
work-group barriers. Input and output retain the upstream structure-of-arrays
layout, so quadrature-point reads by neighboring threads are contiguous.

The 64-point case performs 11,768 floating-point operations per element. The
other sizes use the same operation count scaled to their basis and quadrature
dimensions. Timing contains only repeated kernel launches.

```text
./main [--points 8|64|216|512] [--elements N] [--iters N] [--warmup N]
```

`make run` executes the four real sizes at their default element counts:

```text
./main --points 8
./main --points 64
./main --points 216
./main --points 512
```

The input fields are deterministic synthetic data, so the kernels can be
benchmarked without MFEM, MPI, or a mesh file. CUDA and SYCL both agreed with
the independent host reference within `5e-15`.

The source derived from Laghos is distributed under its BSD-2-Clause license;
see `LICENSE.Laghos` and `NOTICE.Laghos`.
