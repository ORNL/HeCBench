# moe-sum-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME moe-sum
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
