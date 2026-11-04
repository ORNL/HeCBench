# atan2-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME atan2
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
