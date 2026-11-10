# sa-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sa
    MODEL hip
    SOURCES main.cu
    CATEGORIES bioinformatics
)
