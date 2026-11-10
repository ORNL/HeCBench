# shuffle-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME shuffle
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
