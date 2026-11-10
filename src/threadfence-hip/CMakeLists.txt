# threadfence-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME threadfence
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
