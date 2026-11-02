# stiff-ode-hip

A stiff chemistry step at the size of the detailed n-heptane mechanism (560 species, 2500 reactions), after the optimizations in [Performance Profiling on AMD GPUs – Part 5](https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/ai-assist-optimization/README.html). AMD did not publish that kernel; this is not it.

Each zone is an isothermal, constant-density reactor. The advance is backward Euler with an analytic Jacobian and a dense 560×560 LU with partial pivoting. Temperature is fixed per zone, so the Arrhenius coefficients and \(K_c\) are write-once. Each step assembles and factors \(I - \Delta t\,J\) on its first Newton iteration and reuses the factors for the rest of the step (modified Newton), so later iterations reassemble only \(\omega\).

The reaction table is an element-balanced isomer network with the published species and reaction counts (`../stiff-ode-cuda/workload.h`). Every reaction stays inside one element family, so the analytic Jacobian is sparse; the solver stores that matrix and factors it with a dense LU. Its chemistry is not meant to be representative. The rate code still handles third-body, Lindemann, and Troe reactions.

The workload, mechanism size, and host reference (`workload.h`, `mechanism.h`, `reference.h`) live in `../stiff-ode-cuda/` and are shared by the HIP, CUDA, SYCL, and OpenMP ports.

## Kernel

One zone is one block. The kernel uses AMD `readlane`/`readfirstlane` builtins, so this port targets AMD GPUs; `stiff-ode-cuda` is the NVIDIA counterpart.

- Split assembly has one thread per reaction store that reaction's rate terms in global scratch. Then one wave per species row adds them into its columns in reaction-index order, so each row is written once, as \(I - \Delta t\,J\). Waves take rows from an LDS counter, since rows differ widely in cost.
- J (2.5 MB) stays in global memory and is factored in panels. Each panel is factored a sub-panel at a time in an LDS tile of all 560 rows at an odd stride (one thread per row, so the rows fall in distinct banks). Each sub-panel's columns are applied to the rest of the panel, and then the columns right of the panel take all of the panel's updates in one pass, with their U entries in registers and L staged through the tile. Wave 0 finds each pivot with a shuffle butterfly that keeps the serial tie-break.
- The substitution is blocked by 32 rows. Wave 0 solves each diagonal block with `readlane` broadcasts, and the block applies it to the other rows.

`solver_kernel<Block, Panel, Sub>` declares `__launch_bounds__(Block)`. The driver picks the shape by wavefront width: 64-wide (CDNA) runs 1024 threads with 24-column panels and 12-column sub-panels (a 58240 B tile), and 32-wide (RDNA, gfx1250) runs 512 threads with 40-column panels and 8-column sub-panels (40320 B). Odd panel widths are slow on CDNA. The driver checks the tile against `sharedMemPerBlockOptin`.

The kernel keeps the serial operation order of the host reference, which runs the same modified Newton and LU on the CPU with the zones spread over OpenMP threads.

## Build and run

```bash
make run                  # 100 steps, 5 warmup launches, 10 timed launches
```

```text
./main [--zones N] [--iters N] [--steps N] [--warmup N] [--dt SEC]
```

The program checks one launch of `--steps` backward-Euler steps from the initial state against the host reference at 1e-12 relative tolerance and returns if that check fails. It then times the kernel over `--warmup` untimed and `--iters` timed launches. The host reference finishes before the timing, so its OpenMP threads stay off the measurement. Times are per launch, from `std::chrono::steady_clock` around the timed launches and the device synchronize that follows them, so they include launch overhead. The default zone count is one per resident block, from the occupancy API (one per CU).

ROCm 7.1.1. The table below is a shorter run: 2 steps per launch, 2 warmup and 10 timed launches, one zone per CU. `./main` and `make run` use 100 steps, 5 warmup launches, and 10 timed launches.

| GPU | zones | time (s) | max rel. error vs host |
| --- | --- | --- | --- |
| MI210 (gfx90a) | 104 | 0.0212 | 7.4e-16 |

The gfx950 and gfx1100 builds compile; ROCm 7.1.1 has no gfx1250 target.

The OpenMP host reference takes 0.2 to 0.6 s for the same zones on these nodes. A serial one-thread-per-zone GPU kernel took 20 s (MI300A) and 29 s (MI210); it was removed in favor of the host reference.

A rocSOLVER variant (host Newton loop, `rocsolver_dgetrf_strided_batched` and `getrs` over all zones) matched the serial LU to about 1e-15 but was about 1.4× slower than this kernel, with most of its time in `getrf`. It was removed.
