# pingpong-cuda

GPU ping-pong bandwidth. `main-mpi` passes **device pointers** to
`MPI_Send` / `MPI_Recv` (GPU-aware MPI required). `main-nccl` does not.

Use exactly two MPI ranks (one GPU per rank). Run `./main-mpi` only for the
tests below; `make run` also launches `main-nccl`.

Rebuild against the MPI you launch with. Mixing an NVHPC-linked binary with
distro `mpirun` (or the reverse) is not a valid test.

## GPU-aware MPI test (must pass)

Link and launch with the CUDA-aware MPI from the NVIDIA HPC SDK (Makefile
`MPI_ROOT` / `LAUNCHER`):

```bash
make clean
make ARCH=sm_90
# Use the same MPI the binary was linked against, e.g. Makefile LAUNCHER:
/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/mpi/bin/mpirun --mca coll ^hcoll -n 2 ./main-mpi
```

**Pass:** no probe error on stderr; rank 0 prints
`MPI : Transfer size (B): ... Bandwidth (GB/s): ...` from 512 KiB through 1 GiB.

Startup probes GPU-aware MPI (2 doubles, 64 KiB, and the first timed size)
and checks that **device** memory was updated, not only that MPI returned.

## Non-GPU-aware MPI test (must fail quickly)

Rebuild against a **host-only** MPI, then run **that** `mpirun`:

```bash
make clean
make ARCH=sm_90 MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi
/usr/bin/mpirun -n 2 ./main-mpi
```

**Pass for this test:** exit within about 30 s with a stderr error, such as

- `GPU-aware MPI probe failed` — MPI returned, device buffers were wrong
- `GPU-aware MPI probe timed out` — MPI hung on a device pointer; a watchdog aborts

It must not block in `MPI_Recv` indefinitely.
