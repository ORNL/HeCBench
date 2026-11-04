# snake-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME snake
    MODEL omp
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
