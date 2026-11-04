# matrixT-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME matrixT
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
