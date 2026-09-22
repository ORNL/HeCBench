# pingpong-sycl

GPU ping-pong bandwidth. `main-mpi` passes **device pointers** to
`MPI_Send` / `MPI_Recv` (GPU-aware MPI required). `main-ccl` does not.

Use exactly two MPI ranks (one GPU per rank). Run `./main-mpi` only for the
tests below; `make run` also launches `main-ccl`.

Match the SYCL backend, the MPI you **link**, and the launcher. Do not mix
`make HIP=yes` with the Makefile default Intel `MPI_ROOT`.

## GPU-aware MPI test (must pass)

HIP backend and Cray MPICH (link `libmpi_gtl_hsa`):

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
export ONEAPI_DEVICE_SELECTOR=hip:gpu
srun -n 2 ./main-mpi
```

Intel GPU / Level Zero and Intel MPI (Makefile `MPI_ROOT`):

```bash
make clean
make
/opt/intel/oneapi/2024.1/bin/mpirun -n 2 ./main-mpi
```

NVIDIA GPU and CUDA-aware MPI (`CUDA=yes`, NVHPC MPI — see Makefile comments):

```bash
make clean
make CUDA=yes CUDA_ARCH=sm_90
# launch with the NVHPC mpirun used at link time
```

**Pass:** no probe error on stderr; rank 0 prints
`MPI: Transfer size (B): ... Bandwidth (GB/s): ...` from 512 KiB through 1 GiB.

Startup probes GPU-aware MPI (2 doubles, 64 KiB, and the first timed size)
and checks that **device** memory was updated, not only that MPI returned.

## Non-GPU-aware MPI test (must fail quickly)

Rebuild against host-only MPI, or disable GPU support on Cray:

```bash
export MPICH_GPU_SUPPORT_ENABLED=0
srun -n 2 ./main-mpi
```

```bash
make clean
make MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi
/usr/bin/mpirun -n 2 ./main-mpi
```

**Pass for this test:** exit within about 30 s with a stderr error, such as

- `GPU-aware MPI probe failed` — MPI returned, device buffers were wrong
- `GPU-aware MPI probe timed out` — MPI hung on a device pointer; a watchdog aborts

It must not block in `MPI_Recv` indefinitely.
