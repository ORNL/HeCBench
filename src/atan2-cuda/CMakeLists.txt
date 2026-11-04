# atan2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME atan2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    COMPILE_OPTIONS --expt-relaxed-constexpr
)
