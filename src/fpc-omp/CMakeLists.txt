# fpc-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME fpc
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
