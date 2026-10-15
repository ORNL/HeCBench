# stiff-ode-omp

OpenMP target-offload port of `src/stiff-ode-hip`: a backward-Euler chemistry step at the size of the detailed n-heptane mechanism (560 species, 2500 reactions), with an analytic Jacobian, modified Newton, and a dense 560×560 LU with partial pivoting. The reaction table is the element-balanced isomer network in `../stiff-ode-cuda/workload.h`. Each reaction is a first-order edge inside one element family, so the analytic Jacobian is sparse; the solver stores that matrix and factors it with a dense LU. The host reference `../stiff-ode-cuda/reference.h`, shared with the HIP, CUDA, and SYCL ports, runs the same modified Newton and LU on the CPU with the zones spread over OpenMP threads.

One zone is one team. The kernel does split assembly (storing each row as \(I - \Delta t\,J\)), then factors J (global memory) in panels, a sub-panel at a time, through a team-shared tile of all 560 rows at an odd stride, and substitutes team-wide. The trailing columns take each panel's updates in one pass with their U entries in registers. The shape follows the wavefront width (`SolverShapeFor`, the same as HIP and SYCL): 64-wide runs 1024 threads with 24-column panels and 12-column sub-panels (a 58240 B tile), and 32-wide runs 512 threads with 40-column panels and 8-column sub-panels (40320 B).

## Build and run

```bash
make run                  # icpx -fiopenmp -fopenmp-targets=spir64
make -f Makefile.aomp run # ROCm clang, ARCH=gfx942
make -f Makefile.nvc run  # nvc++ -mp=gpu, SM=cc80
```

`make run` checks one launch, then times the kernel. `STEPS` (default 100), `WARMUP` (default 5), and `ITERS` (default 10) are the step count and the untimed and timed launch counts. `Makefile.aomp` is the ROCm/AOMP clang build (`ARCH`, default `gfx942`). `Makefile` is the Intel `icpx -fiopenmp -fopenmp-targets=spir64` build. `Makefile.nvc` is `nvc++ -mp=gpu` (`SM`, default `cc80`; `SM=cc80,cc90` covers A100 and H100), after `module load nvhpc`. All three honor `EXTRA_CFLAGS`. The device query uses HSA on the AOMP build, CUDA on the NVHPC build, then KFD sysfs. If none of those report a device, the program uses the OpenMP default device with a 32-wide wavefront and a compute-unit count of 0, and exits unless `--zones N` is passed.

```text
./main [--zones N] [--iters N] [--steps N] [--warmup N] [--dt SEC]
```

The program checks one launch of `--steps` steps from the initial state against the host reference at 1e-12 relative tolerance and returns if that check fails. It then times the kernel over `--warmup` untimed and `--iters` timed launches. The host reference finishes before the timing. Times are per launch, from `std::chrono::steady_clock` around the synchronous launches, so they include launch overhead. OpenMP has no occupancy query; the tile allows one team per CU, so the default zone count is the CU count.

## OpenMP mapping

`kinetics.h` and `main.cpp` keep the HIP function names, variables, and operation order. A launch is `target teams num_teams(zones)` with a directly nested `parallel num_threads(...)` (SPMD). The team size is `omp_get_num_threads()`, and `#pragma omp barrier` stands in for `__syncthreads()`. The LU tile, the pivot lane partials, and the scalars are team-shared. Clang (AOMP) takes them as `static` arrays with `#pragma omp allocate(...) allocator(omp_pteam_mem_alloc)`. icpx ignores that allocator on a static and gives each thread its own copy, so the pivot tile is not shared and a later team barrier does not complete. On icpx, `launch_solver` allocates the tile with `omp_alloc(omp_pteam_mem_alloc)` in the teams region and the run fails with `team allocation failed` if that returns null. OpenMP may field fewer threads than requested; every loop strides by the team size, so any whole number of waves gives the same result, and a partial wave fails the run.

- `thread_limit` must be a compile-time constant, and it fixes the register budget, so `launch_solver<Ws>` is a template with `thread_limit(SolverShapeFor<Ws>::block)`, the analog of HIP's `__launch_bounds__`.
- NVHPC has no `allocate` directive. Under `__NVCOMPILER` the team arrays are declared in the `teams` region outside the `parallel` region, which NVHPC places in CUDA shared memory only up to 48 KB; the 32-wide tile (40320 B) fits. NVHPC sizes teams by register use: it fields 224 of 512 threads by default and 480 with `maxregcount:128`, which `Makefile.nvc` sets.
- There is no portable wavefront shuffle. The pivot search has the lanes of the first wave publish their best row to the team-shared lane buffer; after a barrier every thread folds those partials, giving the same row as the serial scan and the HIP butterfly. Split assembly takes the wave width as a template argument, so each lane's column accumulators stay in registers. Its waves stride the rows; HIP, CUDA, and SYCL take rows from a shared counter, which needs a lane broadcast.
- The triangular solve stays team-wide, one column per barrier. HIP, CUDA, and SYCL run it on one wave and broadcast each solved entry across the lanes; OpenMP has no lane broadcast or wave-level barrier. A blocked solve (thread 0 substitutes inside each diagonal block, then the team applies the block) cut the barriers to two per block but was slower at every block size from 8 to 64.

## Results

The table below is a shorter run: 2 steps per launch, 2 warmup and 10 timed launches, one zone per CU. `./main` and `make run` use 100 steps, 5 warmup launches, and 10 timed launches.

| GPU | compiler | zones | time (s) | max rel. error vs host |
| --- | --- | --- | --- | --- |
| MI210 (gfx90a) | ROCm 7.1.1 clang | 104 | 0.0351 | 7.4e-16 |
| H100 PCIe (sm_90) | NVHPC 25.9 | 114 | 0.0386 | 3.6e-15 |

The AOMP gfx950 and gfx1100 builds compile.
