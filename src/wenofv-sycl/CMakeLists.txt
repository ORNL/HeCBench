# wenofv-sycl/CMakeLists.txt
#
# -ffp-contract=off keeps device and host FP contraction identical so the
# CPU reference (reference.h) and the SYCL kernels (kernels.h) agree.

add_hecbench_benchmark(
    NAME wenofv
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
    COMPILE_OPTIONS -ffp-contract=off
)
