# ntt-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME ntt
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
