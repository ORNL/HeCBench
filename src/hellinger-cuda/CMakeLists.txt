# hellinger-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hellinger
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
