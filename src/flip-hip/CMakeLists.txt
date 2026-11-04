# flip-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME flip
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
