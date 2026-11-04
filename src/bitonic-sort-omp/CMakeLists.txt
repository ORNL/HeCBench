# bitonic-sort-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitonic-sort
    MODEL omp
    SOURCES main.cpp
    CATEGORIES sorting
)
