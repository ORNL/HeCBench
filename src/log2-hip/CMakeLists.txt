# log2-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME log2
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
