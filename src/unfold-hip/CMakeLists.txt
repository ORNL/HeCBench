# unfold-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME unfold
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
