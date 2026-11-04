# determinant-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME determinant
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipsolver
)
