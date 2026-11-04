# interleave-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME interleave
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
