# hungarian-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hungarian
    MODEL hip
    SOURCES main.cu
    CATEGORIES graph
)
