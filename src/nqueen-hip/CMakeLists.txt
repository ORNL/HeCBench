# nqueen-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME nqueen
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
