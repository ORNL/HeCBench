# stiff-ode-cuda

CUDA port of `src/stiff-ode-hip`: a backward-Euler chemistry step at the size of the detailed n-heptane mechanism (560 species, 2500 reactions), with an analytic Jacobian, modified Newton, and a dense 560×560 LU with partial pivoting. The reaction table is the element-balanced isomer network in `workload.h`. Each reaction is a first-order edge inside one element family, so the analytic Jacobian is sparse; the solver stores that matrix and factors it with a dense LU. `mechanism.h`, `workload.h`, and the host reference `reference.h` live here and are shared with the HIP, SYCL, and OpenMP ports.

One zone is one block. `kinetics.cuh` keeps the HIP function names and operation order. The kernel runs a 512-thread block:

- Split assembly gives each reaction to one thread, then each species row to one warp. Warps take rows from a shared-memory counter, since rows differ widely in cost, and each row is stored as \(I - \Delta t\,J\).
- J (global memory) is factored in 40-column panels. Each panel is factored 8 columns at a time in a 40320 B shared-memory tile (560 rows at an odd stride of 9 doubles, one thread per row, so the rows fall in distinct banks). The columns right of the panel then take all 40 updates in one pass, with their U entries in registers and L staged through the tile.
- The substitution is blocked by 32 rows. Warp 0 solves each diagonal block with `__shfl_sync` broadcasts, and the block applies it to the other rows.

`solver_kernel<Block, Panel, Sub>` declares `__launch_bounds__(Block)`, so the 40 U registers per thread fit. The host reference runs the same modified Newton and LU on the CPU with the zones spread over OpenMP threads. The driver checks the tile against `sharedMemPerBlockOptin` and sets the dynamic shared-memory size with `cudaFuncSetAttribute` before launch.

## Build and run

```bash
make run                  # 100 steps, 5 warmup launches, 10 timed launches
```

`ARCH` overrides the architecture (default `sm_80`, `sm_70` or newer).

```text
./main [--zones N] [--iters N] [--steps N] [--warmup N] [--dt SEC]
```

The program checks one launch of `--steps` steps from the initial state against the host reference at 1e-12 relative tolerance and returns if that check fails. It then times the kernel over `--warmup` untimed and `--iters` timed launches. The host reference finishes before the timing, so its OpenMP threads stay off the measurement. Times are per launch, from `std::chrono::steady_clock` around the timed launches and the device synchronize that follows them, so they include launch overhead. The default zone count is one per resident block (one per SM).

CUDA 13.0. The table below is a shorter run: 2 steps per launch, 2 warmup and 10 timed launches, one zone per SM. `./main` and `make run` use 100 steps, 5 warmup launches, and 10 timed launches.

| GPU | zones | time (s) | max rel. error vs host |
| --- | --- | --- | --- |
| H100 PCIe (sm_80 binary) | 114 | 0.0123 | 7.4e-16 |
