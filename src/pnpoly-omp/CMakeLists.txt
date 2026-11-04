# pnpoly-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME pnpoly
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
