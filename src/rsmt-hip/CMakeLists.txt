# rsmt-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rsmt
    MODEL hip
    SOURCES main.cu
    CATEGORIES graph
)
