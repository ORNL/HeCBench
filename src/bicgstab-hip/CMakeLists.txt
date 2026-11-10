# bicgstab-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bicgstab
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblas hipsparse
)
