# wedford-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME wedford
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
