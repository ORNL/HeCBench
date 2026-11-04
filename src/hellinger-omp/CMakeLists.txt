# hellinger-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hellinger
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
