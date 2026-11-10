# floydwarshall-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME floydwarshall
    MODEL hip
    SOURCES main.cu
    CATEGORIES graph
)
