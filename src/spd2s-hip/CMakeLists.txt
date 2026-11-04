# spd2s-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME spd2s
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipsparse
)
