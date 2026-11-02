# CUDA MACE pipeline benchmark

This benchmark implements the compute-intensive portion of a HydraGNN MACE
forward pass in CUDA. It rebuilds a periodic neighbor graph and applies four
MACE interaction layers, producing a final FP32 node tensor with shape
`[nodes, 128]`.

The benchmark reports three average times:

1. Neighbor-list reconstruction.
2. Four-layer MACE convolution.
3. Total time, calculated as the sum of the first two measurements.

Model setup, fixture loading, host-to-device transfers, and correctness
validation are performed before timing and are not included.

This is a forward-kernel benchmark, not a complete HydraGNN training step. It
does not include model readout or pooling, force differentiation, backward
propagation, optimizer work, or dataset I/O.

## Input workload

The required fixture is:

```text
../data/hydragnn-sc26/mace_pipeline_capture.macepipe
```

It contains:

- Four molecular/material graphs with 77 nodes.
- Captured topology with 1,286 directed edges.
- Atomic numbers, positions, cells, periodic-boundary flags, and graph
  attributes.
- Weights for the four MACE interaction layers.
- Expected intermediate tensors used for validation.

The fixture is managed by DVC and must be available before running the
benchmark. It is always required: correctness is only checkable against its
reference tensors, and the model parameters it carries are the only ones
available.

### Production workload

The timed workload is always a synthetic batch at the dimensions HydraGNN
actually trains on in the SC26 configuration: 128 graphs, 2,726 atoms, and
45,974 directed edges, about 36x the captured fixture. The captured batch's
checkpoint tensors would run to multiple gigabytes at that size, which is why
the fixture stops at four structures and is used only for validation.

The batch is generated on the host by the `production` namespace in
`mace_pipeline.hpp`, shared verbatim by all four ports so every backend times
the same graph:

- Model parameters are copied from the fixture unchanged. They are per-species
  and per-layer, so they do not depend on how many atoms the batch holds.
- Atomic numbers and graph attributes cycle through the captured values, so the
  element mix is the one the weights were built for.
- Each structure is a periodic cubic cell with rejection-sampled positions held
  at a minimum separation. Fractional coordinates are fixed and the cell edge is
  bisected until the total edge count reaches 45,974 exactly, which works
  because scaling a cell scales every distance and so makes neighbor count
  monotone in the cell edge.
- The topology is built with the same host reference the captured path uses, and
  its hash is compared against the device-generated topology before timing.

The synthetic batch has no reference outputs, so it cannot be validated
numerically. Correctness is therefore always established on the captured fixture
first, and only the timed region uses the larger workload.

## Computation

### Neighbor-list reconstruction

The benchmark reconstructs the graph from positions and periodic cells instead
of using the captured edge list as input. For every destination atom, it:

1. Enumerates source atoms and relevant periodic images.
2. Keeps candidates within a 5 Å cutoff.
3. Selects at most 20 nearest neighbors using the deterministic order
   `(distance², source, shift-x, shift-y, shift-z)`.
4. Builds receiver-grouped edge offsets and physical shift vectors.

One 64-thread CUDA block processes each destination atom. Candidate selection
uses a shared-memory staging pool and rank-based top-20 selection. A
hierarchical 64-bit exclusive scan converts neighbor counts into edge offsets.

Neighbor geometry uses FP64 so distance ties and periodic-image selection remain
stable. Shift vectors are converted to FP32 before entering the MACE pipeline.

### Four-layer MACE convolution

The convolution performs:

1. Species embedding.
2. Edge distance, direction, cutoff, and radial-basis feature construction.
3. Graph conditioning of node features.
4. Four interaction layers containing:
   - Node projections for message passing and residual connections.
   - An edge MLP that generates tensor-product weights.
   - Equivariant tensor products and receiver aggregation.
   - Post-interaction projection and normalization.
   - Species-dependent symmetric contraction.
   - Residual addition, sizing, and conditioning for the next layer.

All timed intermediates remain on the GPU. The timed path uses fused kernels for
embedding lookup, tensor-product aggregation, and scale/reshape operations to
avoid unnecessary intermediate storage and dispatches.

## Correctness checks

Validation runs before warmup and timing:

1. The generated neighbor topology and periodic shifts are compared with the
   captured fixture.
2. The explicit convolution path checks 56 intermediate tensors.
3. The fused timed path is run once and its final `[nodes, 128]` tensor is
   compared with the captured final result.

The model comparison uses an absolute tolerance of `2e-4` and a relative
tolerance of `2e-3`. Timing begins only after all checks pass.

## Timing

Timing uses `std::chrono::steady_clock` with CUDA device synchronization before
and after each repeated region. Each reported phase is the elapsed wall time
divided by `--repeat`.

Warmup runs both neighbor reconstruction and convolution `--warmup` times.
Neighbor reconstruction and convolution are then measured independently. The
reported total is:

```text
total time = neighbor reconstruction time + convolution time
```

Because setup and transfers are excluded, the results represent steady-state
device execution and launch overhead for this fixed workload.

## GEMM implementation

Matrix multiplication uses a persistent cuBLAS handle and `cublasSgemm`.
Operands and results are stored as row-major FP32 matrices; the cuBLAS call
swaps their interpretation to perform the equivalent column-major operation.

## Build and run

Build with the standalone Makefile:

```sh
make
```

Run the benchmark:

```sh
make run
```

Equivalent command:

```sh
./main \
  --fixture ../data/hydragnn-sc26/mace_pipeline_capture.macepipe \
  --warmup 100 \
  --repeat 100
```

Useful Makefile overrides include:

```sh
make CC=/usr/local/cuda/bin/nvcc ARCH=sm_80
make run FIXTURE=/path/to/mace_pipeline_capture.macepipe
```

The benchmark can also be built through the repository's top-level CMake
configuration as the `mace-cuda` target.

## Command-line options

```text
--fixture FILE       Required MACEPIPE fixture
--warmup N           Warmup iterations (default: 1; zero is allowed)
--repeat N           Measured iterations (default: 3; must be positive)
-h, --help           Show usage
```

The program exits with status `0` on success, `1` for setup or runtime errors,
and `2` when numerical validation fails.

## Related implementations

Equivalent HIP, SYCL, and OpenMP target-offload implementations are available
in:

- `mace-hip`
- `mace-sycl`
- `mace-omp`

They use the same fixture, neighbor-selection rules, model stages, validation
criteria, and output shape.
