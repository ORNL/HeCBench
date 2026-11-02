# pingpong-sycl

GPU ping-pong bandwidth. `main-mpi` passes **device pointers** to
`MPI_Send` / `MPI_Recv` (GPU-aware MPI required). `main-ccl` does not.

Use exactly two MPI ranks. Each rank uses visible GPU `rank % count`, so
with one visible GPU both ranks share it. Run `./main-mpi` only for the
tests below; `make run` also launches `main-ccl`.

On a GPU or MPI error, a rank calls `MPI_Abort`. Under Slurm, launch with
`srun --kill-on-bad-exit=1` (`-K`) unless the site sets `KillOnBadExit=1`;
otherwise the other rank can keep waiting in `MPI_Send` / `MPI_Recv`.

Match the SYCL backend, the MPI you **link**, and the launcher. Do not mix
`make HIP=yes` with the Makefile default Intel `MPI_ROOT`.

## How main-mpi decides whether MPI is GPU-aware

Before any device pointer reaches MPI, `main-mpi` picks the buffer type from
the SYCL device vendor (NVIDIA: CUDA, AMD: HIP, Intel: Level Zero) and asks
the MPI library (`../pingpong-cuda/gpu_aware_mpi.h`, shared with
pingpong-cuda and pingpong-hip):

| MPI | Decided by |
|-----|------------|
| Cray MPICH | `MPICH_GPU_SUPPORT_ENABLED=1` (default off), then the MPICH query for the vendor |
| Intel MPI | Intel and NVIDIA GPUs: `I_MPI_OFFLOAD` nonzero (default 0). AMD GPUs: always no |
| MVAPICH2 | `MV2_USE_CUDA` / `MV2_USE_ROCM` (`1` yes, `0` no, unset: cannot tell) |
| MPICH 4.0 and later | `MPIX_Query_cuda_support()`, `MPIX_Query_hip_support()`, or `MPIX_Query_ze_support()` (honor `MPIR_CVAR_ENABLE_GPU`) |
| Open MPI with the CUDA or ROCm extension (`mpi-ext.h`) | `MPIX_Query_cuda_support()` or `MPIX_Query_rocm_support()` |
| Open MPI without the extension, Open MPI on Level Zero, other vendors or MPIs | cannot tell |

If the library reports no support, the program aborts and names what
decided it and how to enable GPU buffers. If the library cannot tell, the
program also aborts unless `MPI_GPU_AWARE=1` is set. Set it only when you
know the MPI is GPU-aware. It does not override a library that reports no
support.

## GPU-aware MPI test (must pass)

HIP backend and Cray MPICH (link `libmpi_gtl_hsa`):

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
export ONEAPI_DEVICE_SELECTOR=hip:gpu
srun -K -n 2 ./main-mpi
```

Intel GPU / Level Zero and Intel MPI (Makefile `MPI_ROOT`).
`I_MPI_OFFLOAD` defaults to 0, which turns GPU buffers off. When Level Zero
GPUs are present, `main-mpi` uses only those (not the OpenCL view of the same
GPU), so both ranks may share one GPU:

```bash
make clean
make
export I_MPI_OFFLOAD=1
/opt/intel/oneapi/2024.1/bin/mpirun -n 2 ./main-mpi
```

NVIDIA GPU and CUDA-aware MPI (`CUDA=yes`, NVHPC MPI — see Makefile comments):

```bash
make clean
make CUDA=yes CUDA_ARCH=sm_90
# launch with the NVHPC mpirun used at link time
```

**Pass:** no error on stderr; rank 0 prints
`MPI: Transfer size (B): ... Bandwidth (GB/s): ...` from 512 KiB through 1 GiB.

## Non-GPU-aware MPI test (must fail immediately)

Rebuild against host-only MPI, or disable GPU support on Cray or Intel MPI:

```bash
export MPICH_GPU_SUPPORT_ENABLED=0
srun -K -n 2 ./main-mpi
```

```bash
make clean
make MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi
/usr/bin/mpirun -n 2 ./main-mpi
```

```bash
# Intel MPI: GPU buffers off (this is the default)
export I_MPI_OFFLOAD=0
/opt/intel/oneapi/2024.1/bin/mpirun -n 2 ./main-mpi
```

**Pass for this test:** the program aborts at startup, before any transfer,
with one of

- `MPI library reports no ... GPU-buffer support (...)`
- `MPI library cannot report ... GPU-buffer support (...)`
