# topk-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME topk
    MODEL cuda
    SOURCES main.cu
    COMPILE_OPTIONS --expt-relaxed-constexpr
    CATEGORIES sorting
)
