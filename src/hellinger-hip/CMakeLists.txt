# hellinger-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hellinger
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
