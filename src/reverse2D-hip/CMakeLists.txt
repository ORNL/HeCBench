# reverse2D-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME reverse2D
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
