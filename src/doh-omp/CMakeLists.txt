# doh-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME doh
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
