# segsort-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME segsort
    MODEL cuda
    SOURCES main.cu
    CATEGORIES sorting
    COMPILE_OPTIONS --expt-relaxed-constexpr
)
