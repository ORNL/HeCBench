# score-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME score
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
