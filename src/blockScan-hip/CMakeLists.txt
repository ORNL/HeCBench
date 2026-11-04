# blockScan-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME blockScan
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
