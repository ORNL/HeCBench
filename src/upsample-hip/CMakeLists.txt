# upsample-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME upsample
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
