# log2-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME log2
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
