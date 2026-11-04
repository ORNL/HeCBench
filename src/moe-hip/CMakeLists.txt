# moe-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME moe
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
