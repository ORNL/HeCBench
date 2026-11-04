# glu-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME glu
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
