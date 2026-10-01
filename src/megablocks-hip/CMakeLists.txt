# megablocks-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME megablocks
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblas hipblaslt
)
