# reverse-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME reverse
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
