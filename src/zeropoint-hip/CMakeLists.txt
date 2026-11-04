# zeropoint-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME zeropoint
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
