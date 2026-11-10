# nw-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME nw
    MODEL hip
    SOURCES nw.cu
    CATEGORIES bioinformatics
)
