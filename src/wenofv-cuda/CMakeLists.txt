# wenofv-cuda/CMakeLists.txt
#
# -fmad=false and -ffp-contract=off keep device and host FP contraction
# identical so the CPU reference and GPU kernels (same source) agree.

add_hecbench_benchmark(
    NAME wenofv
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
    COMPILE_OPTIONS -fmad=false -Xcompiler=-ffp-contract=off
)
