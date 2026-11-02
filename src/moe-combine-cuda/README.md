# moe-combine

Intra-node Mixture-of-Experts combine, the bandwidth-heavy half of the dispatch
plus combine measured by the [MORI EP benchmark](https://github.com/ROCm/mori/blob/main/docs/MORI-EP-BENCHMARK.md).
One MPI rank owns one GPU. The default shape is 4096 tokens, hidden 7168,
top-8, 32 experts per rank, and bf16 expert outputs.

## What is timed

Each token draws `topk` distinct experts from the full pool with a fixed hash
of `(source rank, token)`. That is the balanced-gate baseline: random scores,
then top-k, with the hash standing in for the random draw so every rank and
every run share one route. Expert `e` lives on rank `e / experts_per_rank`.
Measured on this hash at 8 ranks and 4096 tokens, a token's experts land on
5.29 ranks on average, 4.63 of them remote. In that draw every token's experts
occupy more than one rank. Dispatch still sends the token once per destination
rank, as MORI does: if an earlier expert of that token already selected the
rank, the later expert is dropped and combine does not read it. Each destination packs
the kept arrivals in source-rank, token, then top-k order. MORI adds those
kept hidden vectors and reduces the router weights separately.
This benchmark scales each kept vector by the weight of the expert that kept
the slot:

```
out[token, h] = sum_{kept k} weight[token, k] * expert_output[dest_rank, slot, h]
```

Accumulation is fp32. The result is stored as bf16 and checked against a host
reference.

Each rank maps its expert-output buffer into the other ranks with IPC and
loads it directly in the combine kernel (MORI's P2P-read combine). CUDA and
HIP use runtime IPC handles. SYCL uses
`sycl::ext::oneapi::experimental::ipc::memory`. A one-block kernel publishes
this rank's epoch and waits until every rank has published that epoch or a
later one. The combine grid is launched after that kernel, on the same
stream. MPI carries the IPC handle bytes and the host barriers. It does not
move device buffers.

## Bandwidth

Reported the way the MORI C++ bench reports it, in GB/s = bytes / 1e9 / seconds.
The time is the slowest rank, and the byte counts are that same rank's. Equal
times keep the rank with more remote reads, then more algo bytes, then the
lower rank. On this hash the per-rank counts differ by about one percent, so
an equal time follows that order.

- **Algo bytes** = `recv_tokens * hidden * sizeof(bf16)`. This is one hidden
  vector per kept token packed at this rank, including tokens that originated
  on the same rank. Measured on this hash at 8 ranks and 4096 tokens, a token
  is kept on 5.29 ranks, so a rank packs about 5.29 hidden vectors per local
  token. MORI's algo bandwidth includes that local traffic.
- **Fabric bytes** = remote reads only, the bytes that cross XGMI or NVLink.
  Measured on this hash at 8 ranks and 4096 tokens, a token is read from 4.63
  remote ranks on average. Experts that share a rank are one read, so fabric
  bytes count the remote kept tokens.

## Build and run

```bash
# one GPU, no peer traffic
make
./main --tokens 128 --hidden 1024 --iters 100 --warmup 100

# one rank per GPU
srun -n 2 ./main --tokens 4096 --hidden 7168 --topk 8
```

`hidden` must be a multiple of 8. `topk` is 1, 2, 4, or 8. Warmup and timed iterations both default to 100. A loaded Cray MPICH module
(`CRAY_MPICH_DIR` or `MPICH_DIR`) selects that prefix and `-lmpi_cray`.
Otherwise the build uses OpenMPI. Override either choice with `MPI_ROOT`
and `MPI_LIB`.
