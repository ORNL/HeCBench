# f16atomic-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME f16atomic
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
