# relu-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME relu
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
