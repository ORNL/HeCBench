# matrix-rotate-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME matrix-rotate
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
