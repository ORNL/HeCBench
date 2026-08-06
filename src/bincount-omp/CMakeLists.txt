# bincount-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bincount
    MODEL omp
    SOURCES main.cpp
    CATEGORIES algorithms
)
