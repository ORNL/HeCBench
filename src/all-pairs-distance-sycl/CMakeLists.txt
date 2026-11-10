# all-pairs-distance-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME all-pairs-distance
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
