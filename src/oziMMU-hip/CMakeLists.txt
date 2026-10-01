# oziMMU-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME oziMMU
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES hipblas
)
