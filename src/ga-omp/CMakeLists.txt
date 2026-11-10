# ga-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME ga
    MODEL omp
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
