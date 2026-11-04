# gaussian-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gaussian
    MODEL hip
    SOURCES gaussianElim.cu utils.cu
    CATEGORIES math
)
