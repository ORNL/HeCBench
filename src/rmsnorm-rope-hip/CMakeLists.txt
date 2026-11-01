# rmsnorm-rope-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rmsnorm-rope
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
