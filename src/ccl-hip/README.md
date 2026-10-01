# ccl-hip

A collective-communication (all-reduce) bandwidth benchmark built on **HIP + RCCL + MPI**.
It assigns one GPU per MPI rank (by node-local rank) and measures average transfer time
and bandwidth for `float32` and `bfloat16` over a range of message sizes.

Source: `main.cu` — usage: `./main <repeat>`

## Requirements

- ROCm (HIP runtime + RCCL): `/opt/rocm`
- An MPI implementation. This directory is set up for **Cray MPICH** (see below).

## Building with Cray MPI

This system uses **cray-mpich** rather than the OpenMPI path that is the Makefile default.

- **MPI root:** `/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0`
  (also exposed via the `$MPICH_DIR` / `$CRAY_MPICH_DIR` environment variables)
  - Headers: `$MPI_ROOT/include/mpi.h`
  - Libraries: `$MPI_ROOT/lib/libmpi.so`

Build by pointing `MPI_ROOT` at the Cray MPICH install:

```bash
make MPI_ROOT=/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0
```

Or, since the environment already defines `$MPICH_DIR`:

```bash
make MPI_ROOT=$MPICH_DIR
```

To clean:

```bash
make clean
```

## Running

### Cray runtime libraries

The Cray-linked binary depends on the Cray Compiler Environment (CCE) runtime libraries
(`libmodules.so.1`, `libfi.so.1`, `libcraymath.so.1`, `libf.so.1`, `libu.so.1`,
`libcsup.so.1`), pulled in through `libmpi_cray`. Make them available on the loader path.
The MPICH build targets Cray 20.0, so use the matching CCE 20.0 libs:

```bash
export LD_LIBRARY_PATH=/opt/cray/pe/cce/20.0.0/cce/x86_64/lib:$LD_LIBRARY_PATH
```

Verify no missing dependencies with `ldd ./main` (there should be no `not found` lines).

### Multi-rank launch (Slurm)

On a Cray/Slurm system the launcher is **`srun`** (there is no `mpirun`), and it must run
inside a Slurm allocation. With one rank per GPU (e.g. 2 ranks on a node with >= 2 GPUs):

```bash
srun -n 2 ./main 50
```

> Note: `srun` requires an available node allocation. If the queue reports
> `Required node not available (down, drained or reserved)` the job will be queued/revoked
> and cannot launch until a node is allocatable.

### Single-rank (no allocation)

If Slurm cannot allocate a node but the current node has GPUs directly accessible, the
benchmark also runs as a single MPI rank (singleton) — RCCL builds a 1-rank communicator:

```bash
export LD_LIBRARY_PATH=/opt/cray/pe/cce/20.0.0/cce/x86_64/lib:$LD_LIBRARY_PATH
./main 50
```

## Expected output

```
  MPI rank 0 assigned to GPU device 0

=== Data type: float32 ===
Transfer size (B):    4194304, Average Transfer Time (s):     0.000009025, Bandwidth (GB/s):   432.85
MPI Rank 0: PASS
...
=== Data type: bfloat16 ===
...
MPI Rank 0: PASS
```

## Makefile note

The Makefile default `MPI_ROOT` is an OpenMPI Debian path
(`/usr/lib/x86_64-linux-gnu/openmpi`) and its `LAUNCHER` default is `/usr/bin/mpirun -n 2`.
Neither is valid on this Cray system, so override `MPI_ROOT` at build time as shown above
and launch with `srun` instead of `make run`.
