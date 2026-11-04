# spsm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME spsm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipsparse
)
