# all-pairs-distance-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME all-pairs-distance
    MODEL hip
    SOURCES main.cu
    CATEGORIES bioinformatics
)
