# ga-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ga
    MODEL hip
    SOURCES main.cu
    CATEGORIES bioinformatics
)
