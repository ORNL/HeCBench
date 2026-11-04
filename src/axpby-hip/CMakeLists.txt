# axpby-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME axpby
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
