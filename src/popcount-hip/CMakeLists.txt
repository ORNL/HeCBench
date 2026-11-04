# popcount-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME popcount
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
