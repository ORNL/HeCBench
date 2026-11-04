# epistasis-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME epistasis
    MODEL omp
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
