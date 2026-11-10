# winograd-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME winograd
    MODEL hip
    SOURCES main.cu utils.cu
    CATEGORIES ml
)
