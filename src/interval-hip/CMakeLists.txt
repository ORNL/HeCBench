# interval-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME interval
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
