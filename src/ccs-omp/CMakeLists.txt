# ccs-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME ccs
    MODEL omp
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
