# wenofv-hip/CMakeLists.txt
#
# -ffp-contract=off keeps device and host FP contraction identical so the
# CPU reference and GPU kernels (same source) agree.

add_hecbench_benchmark(
    NAME wenofv
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffp-contract=off
)
