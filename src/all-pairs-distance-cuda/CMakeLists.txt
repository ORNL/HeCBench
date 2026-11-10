# all-pairs-distance-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME all-pairs-distance
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bioinformatics
)
