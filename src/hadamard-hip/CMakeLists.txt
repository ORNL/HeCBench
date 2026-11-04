# hadamard-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hadamard
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
