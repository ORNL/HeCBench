# spmm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME spmm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipsparse
)
