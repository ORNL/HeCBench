# groupnorm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME groupnorm
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
