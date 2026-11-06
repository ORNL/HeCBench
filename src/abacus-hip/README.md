# ABACUS kernels (HIP)

HIP port of `abacus-cuda`. One process runs the same spherical-harmonic sizes
and neighbor list, with the same inputs and checks.

```sh
./abacus [iterations]
./abacus <vector-count> <maximum-l> <particle-count> <cutoff> <iterations>
```

`make run` uses the Si16, Pt, and Fe example sizes, then 262144 particles at
cutoff 0.04. The five-argument form runs one spherical-harmonic size and one
neighbor list. Both spherical-harmonic kernels are checked against
`abacus-cuda/reference.cpp` before timing. A failed check skips the timed
launches for that size. Both neighbor-list results are checked for every
particle against the host count of every cell that can hold a point inside
the cutoff. The first 512 particles are also checked all-pairs, before
timing. A failed check skips those timed
launches. Timed kernels use `std::chrono`. The process prints `ABACUS: PASS`
when every check matches.
