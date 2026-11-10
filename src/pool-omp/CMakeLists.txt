# pool-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME pool
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
