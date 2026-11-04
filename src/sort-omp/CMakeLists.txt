# sort-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME sort
    MODEL omp
    SOURCES main.cpp
    CATEGORIES sorting
)
