# schur-complement-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME schur-complement
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
