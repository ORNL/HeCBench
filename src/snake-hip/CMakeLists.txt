# snake-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME snake
    MODEL hip
    SOURCES main.cu
    CATEGORIES bioinformatics
)
