# gels-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gels
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    COMPILE_OPTIONS -DHIPBLAS_V2
    LINK_LIBRARIES hipblas
)
