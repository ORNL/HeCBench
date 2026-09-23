# HydraGNN SC26 real-workload fixtures

These fixtures were captured from ORNL/HydraGNN commit
`d251f81627e6677c8636babe52ec0b020b2d27b7` using the published
`examples/multidataset_hpo_sc26/gfm_mlip.json` configuration (MACE, four
interaction layers, 128 hidden channels, FP64, 5 Å cutoff, 20-neighbor cap).

The source records are public SC26-listed datasets:

- `colabfit/ANI-1x`, rows 0 and 1 for operation-boundary captures
- `nimashoghi/mptrj`, rows 0 and 1 for operation-boundary captures

`capture_manifest.json` records exact dimensions and upstream identifiers.
`SHA256SUMS` records artifact hashes.

Files:

- `mace_pipeline_capture.macepipe`: version-2 unified four-structure fixture
  containing raw graph offsets, positions, cells, and PBC flags for topology
  reconstruction, model state, and 56 convolution checkpoint references.

The four-structure pipeline capture is a correctness fixture. Its tensor values
are real runtime values; larger performance profiles use the measured
full-batch dimensions to avoid storing multi-gigabyte tensors.
