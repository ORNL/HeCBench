# kiss-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME kiss
    MODEL hip
    SOURCES main.cu
    CATEGORIES encoding
)
