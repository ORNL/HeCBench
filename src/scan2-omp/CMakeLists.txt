# scan2-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME scan2
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
