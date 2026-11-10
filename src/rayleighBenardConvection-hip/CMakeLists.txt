# rayleighBenardConvection-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rayleighBenardConvection
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
    LINK_LIBRARIES hipblas
)
