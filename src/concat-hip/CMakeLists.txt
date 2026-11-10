# concat-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME concat
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
