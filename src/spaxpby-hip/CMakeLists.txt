# spaxpby-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME spaxpby
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipsparse
)
