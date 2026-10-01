# ccl-cuda

NCCL collective-communication benchmark. Each MPI rank drives one GPU and runs
`ncclAllReduce` over a range of message sizes, reporting average transfer time
and bandwidth for `float32` and `bfloat16` data.

## Requirements

- NVIDIA HPC SDK (provides `nvcc`, MPI/HPC-X, and NCCL)
- One GPU per MPI rank

## Configuration

Paths are set at the top of the `Makefile` and assume an NVIDIA HPC SDK layout.
Point `NVHPC_SDK` at your installation; the MPI and NCCL locations are derived
from it. All four use `?=`, so an exported environment variable takes precedence
over the default:

```make
NVHPC_SDK ?= /opt/nvidia/hpc_sdk/Linux_x86_64/26.5
MPI_ROOT  ?= $(NVHPC_SDK)/comm_libs/13.2/hpcx/hpcx-2.50/ompi4
NCCL_ROOT ?= $(NVHPC_SDK)/comm_libs/nccl
LAUNCHER  ?= $(NVHPC_SDK)/comm_libs/mpi/bin/mpirun -n 2
```

Adjust the HPC-X / NCCL sub-paths if your SDK version differs.

## Build

Set `ARCH` to your GPU's compute capability (e.g. `sm_80` for Ampere, `sm_90`
for Hopper). If your SDK lives elsewhere, override `NVHPC_SDK` on the command
line:

```bash
make ARCH=sm_80 NVHPC_SDK=/path/to/Linux_x86_64/26.5
```

## Run

The `run` target launches the benchmark across 2 ranks with a repeat count of 50:

```bash
make run ARCH=sm_80 NVHPC_SDK=/path/to/Linux_x86_64/26.5
```

To run the binary directly, make sure the NCCL and MPI runtime libraries are on
`LD_LIBRARY_PATH`:

```bash
LD_LIBRARY_PATH=/path/to/Linux_x86_64/26.5/comm_libs/nccl/lib:/path/to/Linux_x86_64/26.5/comm_libs/13.2/hpcx/hpcx-2.50/ompi4/lib:$LD_LIBRARY_PATH \
  /path/to/Linux_x86_64/26.5/comm_libs/mpi/bin/mpirun -n 2 ./main 50
```

The single argument (`50` above) is the number of repetitions per message size.

## Notes

- Message size sweeps per-buffer element counts from 1M up to 1,048,576,000
  (`1000 * 1024 * 1024`) elements, stepping by 10x.
- The reported transfer size is `sizeof(dtype) * elements * num_ranks`, so it
  scales with the number of ranks (e.g. ~7.81 GiB for `float32` and ~3.91 GiB
  for `bfloat16` at 2 ranks).
