# dwconv-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME dwconv
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
