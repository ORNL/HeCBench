# zeropoint-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME zeropoint
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
