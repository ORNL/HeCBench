# divergence-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME divergence
    MODEL cuda
    SOURCES divergence.cu
    CATEGORIES math
)
