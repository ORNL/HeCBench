# conversion-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME conversion
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
