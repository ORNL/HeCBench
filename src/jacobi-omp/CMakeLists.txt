# jacobi-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME jacobi
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
