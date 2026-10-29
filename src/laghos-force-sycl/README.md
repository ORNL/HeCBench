# laghos-force-sycl

SYCL 2020 port of the Laghos 3D partial-assembly force-action benchmark. It
preserves the `force_mult_3d` function, parameter names, `--points` sizes, tensor layouts, work-group size, contraction order,
local-memory intermediates, and four barriers. See
`../laghos-force-cuda/README.md` for the workload, optimization, command-line
interface, upstream provenance, and license.
