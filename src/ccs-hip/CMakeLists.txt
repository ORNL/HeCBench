# ccs-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ccs
    MODEL hip
    SOURCES main.cu
    CATEGORIES bioinformatics
)
