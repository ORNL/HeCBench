# mace-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mace
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
    LINK_LIBRARIES rocblas
)
