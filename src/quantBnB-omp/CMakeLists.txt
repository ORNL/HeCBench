# quantBnB-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME quantBnB
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
