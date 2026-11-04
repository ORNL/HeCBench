# deredundancy-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME deredundancy
    MODEL omp
    SOURCES main.cpp utils.cpp
    CATEGORIES bioinformatics
)
