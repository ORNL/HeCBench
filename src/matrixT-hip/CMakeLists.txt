# matrixT-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME matrixT
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
