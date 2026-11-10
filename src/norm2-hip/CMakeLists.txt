# norm2-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME norm2
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblas
)
