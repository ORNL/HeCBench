# pingpong-cuda

GPU ping-pong bandwidth. `main-mpi` passes **device pointers** to
`MPI_Send` / `MPI_Recv` (GPU-aware MPI required). `main-nccl` does not.

Use exactly two MPI ranks. Each rank uses visible GPU `rank % count`, so
with one visible GPU both ranks share it. Run `./main-mpi` only for the
tests below; `make run` also launches `main-nccl`.

On a GPU or MPI error, a rank calls `MPI_Abort`. Under Slurm, launch with
`srun --kill-on-bad-exit=1` (`-K`) unless the site sets `KillOnBadExit=1`;
otherwise the other rank can keep waiting in `MPI_Send` / `MPI_Recv`.

Rebuild against the MPI you launch with. Mixing an NVHPC-linked binary with
distro `mpirun` (or the reverse) is not a valid test.

## How main-mpi decides whether MPI is GPU-aware

Before any device pointer reaches MPI, `main-mpi` asks the MPI library
(`gpu_aware_mpi.h`, shared with pingpong-hip and pingpong-sycl):

| MPI | Decided by |
|-----|------------|
| Cray MPICH | `MPICH_GPU_SUPPORT_ENABLED=1` (default off), then `MPIX_Query_cuda_support()` |
| Intel MPI | `I_MPI_OFFLOAD` nonzero (default 0) |
| MVAPICH2 | `MV2_USE_CUDA` (`1` yes, `0` no, unset: cannot tell) |
| MPICH 4.0 and later | `MPIX_Query_cuda_support()` (honors `MPIR_CVAR_ENABLE_GPU`) |
| Open MPI with the CUDA extension (`mpi-ext.h`) | `MPIX_Query_cuda_support()` |
| Anything else | cannot tell |

If the library reports no support, the program aborts and names what
decided it and how to enable GPU buffers. If the library cannot tell, the
program also aborts unless `MPI_GPU_AWARE=1` is set. Set it only when you
know the MPI is GPU-aware. It does not override a library that reports no
support.

## GPU-aware MPI test (must pass)

Link and launch with the CUDA-aware MPI from the NVIDIA HPC SDK (Makefile
`MPI_ROOT` / `LAUNCHER`):

```bash
make clean
make ARCH=sm_90
# Use the same MPI the binary was linked against, e.g. Makefile LAUNCHER:
/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/mpi/bin/mpirun --mca coll ^hcoll -n 2 ./main-mpi
```

**Pass:** no error on stderr; rank 0 prints
`MPI : Transfer size (B): ... Bandwidth (GB/s): ...` from 512 KiB through 1 GiB.

## Non-GPU-aware MPI test (must fail immediately)

Rebuild against a **host-only** MPI, then run **that** `mpirun`:

```bash
make clean
make ARCH=sm_90 MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi
/usr/bin/mpirun -n 2 ./main-mpi
```

**Pass for this test:** the program aborts at startup, before any transfer,
with one of

- `MPI library reports no CUDA GPU-buffer support (...)`
- `MPI library cannot report CUDA GPU-buffer support (...)`
