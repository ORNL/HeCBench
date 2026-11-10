# vol2col-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME vol2col
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
