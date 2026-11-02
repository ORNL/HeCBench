# pingpong-hip

GPU ping-pong bandwidth. `main-mpi` passes **device pointers** to
`MPI_Send` / `MPI_Recv` (GPU-aware MPI required). `main-nccl` (RCCL) does not.

Use exactly two MPI ranks. Each rank uses visible GPU `rank % count`, so
with one visible GPU both ranks share it. Run `./main-mpi` only for the
tests below; `make run` also launches `main-nccl`.

On a GPU or MPI error, a rank calls `MPI_Abort`. Under Slurm, launch with
`srun --kill-on-bad-exit=1` (`-K`) unless the site sets `KillOnBadExit=1`;
otherwise the other rank can keep waiting in `MPI_Send` / `MPI_Recv`.

The Makefile default `MPI_ROOT` (`/usr/lib/x86_64-linux-gnu/openmpi`) is often
**host-only**. Rebuild against GPU-aware MPI for the pass test (ROCm-aware
Open MPI, or Cray MPICH plus `libmpi_gtl_hsa`). Launch with the same MPI.

## How main-mpi decides whether MPI is GPU-aware

Before any device pointer reaches MPI, `main-mpi` asks the MPI library
(`../pingpong-cuda/gpu_aware_mpi.h`, shared with pingpong-cuda and
pingpong-sycl):

| MPI | Decided by |
|-----|------------|
| Cray MPICH | `MPICH_GPU_SUPPORT_ENABLED=1` (default off), then `MPIX_Query_hip_support()` |
| Intel MPI | always no (Intel MPI does not support AMD GPU buffers) |
| MVAPICH2 | `MV2_USE_ROCM` (`1` yes, `0` no, unset: cannot tell) |
| MPICH 4.0 and later | `MPIX_Query_hip_support()` (honors `MPIR_CVAR_ENABLE_GPU`) |
| Open MPI 5 with the ROCm extension (`mpi-ext.h`) | `MPIX_Query_rocm_support()` |
| Open MPI 4 (for example 4.1 + ROCm-enabled UCX), anything else | cannot tell |

If the library reports no support, the program aborts and names what
decided it and how to enable GPU buffers. If the library cannot tell, the
program also aborts unless `MPI_GPU_AWARE=1` is set. Set it only when you
know the MPI is GPU-aware. It does not override a library that reports no
support.

## GPU-aware MPI test (must pass)

```bash
# Cray MPICH on AMD GPUs (link -lmpi_gtl_hsa)
export MPICH_GPU_SUPPORT_ENABLED=1
srun -K -n 2 ./main-mpi
```

```bash
# ROCm-aware Open MPI 5 (not the typical distro package)
make clean
make MPI_ROOT=/path/to/rocm-aware-openmpi
/path/to/rocm-aware-openmpi/bin/mpirun -n 2 ./main-mpi
```

```bash
# Open MPI 4.1 + ROCm-enabled UCX: the library cannot report support
MPI_GPU_AWARE=1 /path/to/openmpi-4.1/bin/mpirun -x MPI_GPU_AWARE -n 2 ./main-mpi
```

**Pass:** no error on stderr; rank 0 prints
`MPI : Transfer size (B): ... Bandwidth (GB/s): ...` from 512 KiB through 1 GiB.

## Non-GPU-aware MPI test (must fail immediately)

Same `main-mpi` sources, GPU support off or a host-only MPI **after a rebuild**
against that MPI:

```bash
# Cray MPICH: GPU support off (same GPU-aware-linked binary)
export MPICH_GPU_SUPPORT_ENABLED=0
srun -K -n 2 ./main-mpi
```

```bash
# Distro / host-only Open MPI (Makefile default)
make clean
make MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi
/usr/bin/mpirun -n 2 ./main-mpi
```

**Pass for this test:** the program aborts at startup, before any transfer,
with one of

- `MPI library reports no HIP GPU-buffer support (...)`
- `MPI library cannot report HIP GPU-buffer support (...)`
