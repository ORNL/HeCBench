# f16max-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME f16max
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
