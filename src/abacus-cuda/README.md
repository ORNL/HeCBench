# ABACUS kernels

One benchmark runs ABACUS spherical harmonics and neighbor counting in a
single process. The baseline `spherical_harmonics` kernels are derived from
ABACUS (LGPL-3.0), commit `2d1da7056f74da36c7a3539d3416522e9da6e073`
(`2d1da70`). The other kernels implement the algorithms in that commit.

```sh
./abacus [iterations]
./abacus <vector-count> <maximum-l> <particle-count> <cutoff> <iterations>
```

With no size arguments, or with only an iteration count, the run is:

- spherical harmonics for three ABACUS example sizes
  - Si16: 16625 vectors, maximum l = 1 (`examples/34_gpu/01_pw_Si16`)
  - Pt: 28191 vectors, maximum l = 3 (`examples/28_efield/01_Pt_slab`)
  - Fe: 5473 vectors, maximum l = 4 (`examples/16_uspp/01_pw_Fe`, Q(G) on the density grid)
- neighbor counting for 262144 particles at cutoff 0.04

The five-argument form runs one spherical-harmonic size and one neighbor list.
`make run` uses the example sizes. The default iteration count is 100.

The process prints `ABACUS: PASS` when every spherical-harmonic size and the
neighbor list match their references.

## Spherical harmonics

Real spherical harmonics for a batch of reciprocal-space vectors, following
[`cal_ylm_real`](https://github.com/deepmodeling/abacus-develop/blob/develop/source/source_base/kernels/cuda/math_ylm_op.cu).

The output layout is `(l, m)` in the order `m = 0, +1, -1, +2, -2, ...` for
each `l`.

- `spherical_harmonics` follows the original ABACUS loop structure and
  materializes the associated-Legendre triangle.
- `spherical_harmonics_optimized` iterates by `m` and keeps only the previous
  two Legendre values in registers. It also advances normalization factors and
  `sin(m*phi)`/`cos(m*phi)` by recurrence.

`reference.cpp` is the host reference shared with the HIP, SYCL, and OpenMP
builds. It evaluates explicit Legendre polynomials and does not use either
kernel recurrence. Before timing, every vector from both kernels is checked
against that reference. If the check fails, the timed launches for that size
are skipped. Otherwise both kernels are timed with `std::chrono` around the
launches and one device synchronize, then compared on the device.

## Neighbor list

Uniform cell-list construction and neighbor counting within a cutoff,
following
[`gpu_neighlist_kernels.cu`](https://github.com/deepmodeling/abacus-develop/blob/develop/source/source_cell/kernels/cuda/gpu_neighlist_kernels.cu).

Particles are deterministic points in the unit cube. Cell binning is not
timed.

- `count_neighbors` follows the original per-particle search. Each thread
  gathers positions from the 27 neighboring cells through the cell list.
- `count_neighbors_optimized` packs positions in cell order. A cell with at
  least 16 particles is counted by blocks of 64 that stage each neighbor cell
  in shared memory and reuse it. Smaller cells use one thread per particle
  over that packed list, so a sparse grid does not pay for staging. Packing
  and the block map are outside the timed region.

The cell edge is at least the cutoff, so every pair inside the cutoff falls
in the 27-cell stencil. Before timing, both results are checked for every
particle against a host count of every cell that can hold a point inside the
cutoff, so a short edge fails that check. The first 512 particles are also
checked with an all-pairs count. If either check fails, the timed
launches are skipped. Otherwise both kernels are timed with `std::chrono`
around the launches and one device synchronize, then compared on the device.
