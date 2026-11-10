# nw-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME nw
    MODEL omp
    SOURCES nw.cpp
    CATEGORIES bioinformatics
)
