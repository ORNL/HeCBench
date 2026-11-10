# surfel-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME surfel
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
