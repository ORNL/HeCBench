# nqueen-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME nqueen
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
