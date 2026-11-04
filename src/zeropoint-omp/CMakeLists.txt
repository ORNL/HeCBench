# zeropoint-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME zeropoint
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
