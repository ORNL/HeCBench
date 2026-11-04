# ludb-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ludb
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblas
)
