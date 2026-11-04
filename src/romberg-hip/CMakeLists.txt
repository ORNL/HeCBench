# romberg-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME romberg
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
