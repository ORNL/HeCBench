# jacobi-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME jacobi
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
