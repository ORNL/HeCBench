# moe-combine-hip

HIP build of the multi-GPU MoE combine. `main.cu` is the HIP driver. The
kernel utilities in `combine.cuh` and the host reference in `reference.hpp`
are shared with `moe-combine-cuda`. See that README for the workload.
