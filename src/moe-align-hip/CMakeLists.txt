# moe-align-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME moe-align
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
