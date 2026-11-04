# match-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME match
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
