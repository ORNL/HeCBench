# cross-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME cross
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
