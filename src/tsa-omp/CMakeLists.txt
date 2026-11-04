# tsa-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME tsa
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
