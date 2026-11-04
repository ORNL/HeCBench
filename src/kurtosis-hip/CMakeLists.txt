# kurtosis-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME kurtosis
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
