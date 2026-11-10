# cross-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME cross
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
