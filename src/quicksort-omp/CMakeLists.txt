# quicksort-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME quicksort
    MODEL omp
    SOURCES main.cpp
    CATEGORIES sorting
)
