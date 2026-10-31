# vdi-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME vdi
    MODEL hip
    SOURCES main.cu fused.cu libraries.cu
    CATEGORIES math
    LINK_LIBRARIES hipblas hipsolver pthread
)
