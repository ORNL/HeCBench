# eigenvalue-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME eigenvalue
    MODEL hip
    SOURCES main.cu reference.cu
    CATEGORIES math
)
