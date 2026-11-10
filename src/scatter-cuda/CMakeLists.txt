# scatter-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME scatter
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
    COMPILE_OPTIONS --expt-relaxed-constexpr
)
