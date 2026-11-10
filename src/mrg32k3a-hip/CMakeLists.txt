# mrg32k3a-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mrg32k3a
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hiprand
)
