# prefetch-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME prefetch
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
