# accuracy-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME accuracy
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
