# fwt-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME fwt
    MODEL omp
    SOURCES main.cpp reference.cpp
    CATEGORIES math
)
