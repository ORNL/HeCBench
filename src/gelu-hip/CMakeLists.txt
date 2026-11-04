# gelu-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gelu
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
