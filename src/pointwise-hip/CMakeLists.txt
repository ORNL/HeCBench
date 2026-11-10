# pointwise-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME pointwise
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
