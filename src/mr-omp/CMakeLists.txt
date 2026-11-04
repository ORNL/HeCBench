# mr-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mr
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
