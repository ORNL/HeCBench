# remap-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME remap
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
