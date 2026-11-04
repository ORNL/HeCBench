# langford-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME langford
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
