# match-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME match
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
