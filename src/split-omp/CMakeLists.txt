# split-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME split
    MODEL omp
    SOURCES main.cpp
    CATEGORIES sorting
)
