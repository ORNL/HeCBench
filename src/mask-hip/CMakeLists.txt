# mask-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mask
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
