# blockAccess-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blockAccess
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
