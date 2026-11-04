# quantVLLM-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME quantVLLM
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
    COMPILE_OPTIONS --expt-relaxed-constexpr
)
