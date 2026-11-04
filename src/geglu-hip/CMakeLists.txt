# geglu-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME geglu
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
