# scan-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME scan
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
