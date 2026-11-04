# epistasis-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME epistasis
    MODEL hip
    SOURCES main.cu
    CATEGORIES bioinformatics
)
