# stiff-ode-sycl

SYCL 2020 port of `src/stiff-ode-hip`: a backward-Euler chemistry step at the size of the detailed n-heptane mechanism (560 species, 2500 reactions), with an analytic Jacobian, modified Newton, and a dense 560×560 LU with partial pivoting. The reaction table is the element-balanced isomer network in `../stiff-ode-cuda/workload.h`. Each reaction is a first-order edge inside one element family, so the analytic Jacobian is sparse; the solver stores that matrix and factors it with a dense LU. The host reference (`../stiff-ode-cuda/reference.h`) runs the same modified Newton and LU on the CPU with the zones spread over OpenMP threads.

One zone is one work-group. `kinetics.hpp` keeps the CUDA and HIP function names, algorithm, and operation order:

- Split assembly takes rows from a local-memory counter (`atomic_ref`, then `group_broadcast` across the sub-group) and stores each row as \(I - \Delta t\,J\).
- J (global memory) is factored in panels, a sub-panel at a time, in a local-memory tile of all 560 rows at an odd stride. The trailing columns take each panel's updates in one pass with their U entries in registers. The pivot search uses `sycl::permute_group_by_xor` on sub-group 0.
- The substitution is blocked by 32 rows (at most one sub-group). Sub-group 0 solves each diagonal block, broadcasting with `select_from_group` (`readlane` on AMD).

The shape follows the sub-group width (`SolverShapeFor`): 64-wide runs 1024 items with 24-column panels and 12-column sub-panels (58240 B of local memory), and 32-wide runs 512 items with 40-column panels and 8-column sub-panels (40320 B).

## Build and run

```bash
make run                  # 100 steps, 5 warmup launches, 10 timed launches
```

The default compiler is `/storage/users/zjin/sycl_workspace/llvm-install/sycl-fork-a16dfdf5-cuda12.5-rocm7.1.1/bin/clang++` (DPC++ 7.2.0); override it with `CC=`. `HIP=yes HIP_ARCH=gfx942` builds for AMD and `CUDA=yes CUDA_ARCH=sm_80` for NVIDIA; the default is a SPIR-V build. The kernel is a template on the sub-group width, instantiated for 16, 32, and 64, so the per-lane register arrays and lane arithmetic are sized at compile time. The driver launches the first of 32, 64, and 16 that the device lists in `sub_group_sizes` (64 on CDNA, 32 on NVIDIA). Each instantiation requests its width and `max_work_group_size<SolverShapeFor<Ws>::block>` through compile-time kernel properties (`get(properties_tag)` on the kernel functor); this DPC++ ignores `[[intel::max_work_group_size]]`. The work-group bound is the analog of CUDA's `__launch_bounds__`: without it the register allocation assumes smaller work-groups. The host reference is built with `-Xarch_host -fopenmp` and linked with `-lomp`. On a node with both AMD and NVIDIA GPUs, select the NVIDIA one with `ONEAPI_DEVICE_SELECTOR=cuda:*`.

```text
./main [--zones N] [--iters N] [--steps N] [--warmup N] [--dt SEC]
```

The program checks one launch of `--steps` steps from the initial state against the host reference at 1e-12 relative tolerance and returns if that check fails. It then times the kernel over `--warmup` untimed and `--iters` timed launches. The host reference finishes before the timing, so its OpenMP threads stay off the measurement. Times are per launch, from `std::chrono::steady_clock` around the timed launches and the queue wait that follows them, so they include launch overhead. SYCL has no occupancy query; the tile allows one work-group per CU, so the default zone count is the CU count.

DPC++ 7.2.0 (CUDA 12.5, ROCm 7.1.1 backends). The table below is a shorter run: 2 steps per launch, 2 warmup and 10 timed launches, one zone per CU. `./main` and `make run` use 100 steps, 5 warmup launches, and 10 timed launches.

| GPU | zones | time (s) | max rel. error vs host |
| --- | --- | --- | --- |
| H100 PCIe (sm_80 and sm_90 binaries) | 114 | 0.0138 | 7.4e-16 |
| MI210 (gfx90a) | 104 | 0.0212 | 7.4e-16 |

The gfx950 and gfx1100 builds compile.
