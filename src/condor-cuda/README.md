# condor

[ORNL Condor](https://github.com/ORNL/Condor) (`Calculate_Dense_Temperatures` /
`T_device`) at commit `ca3c4baecbfb04876270b9bb6b180b742eb4a948`.

Condor is a Kokkos/MPI thermal solver for metal additive manufacturing. The
full application also depends on Stork and nlohmann/json, and this checkout
ships no scan-case inputs. HeCBench keeps the kernel that evaluates the
published heat-source integral.


```Usage
./main [--nx N] [--ny N] [--nz N] [--nodes N] [--iters N] [--warmup N] [--block N]
```

The source derived from Condor is distributed under its BSD-3-Clause license;
see `LICENSE`.
