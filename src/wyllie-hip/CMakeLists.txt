# wyllie-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME wyllie
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
