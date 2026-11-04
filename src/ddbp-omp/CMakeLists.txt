# ddbp-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME ddbp
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
