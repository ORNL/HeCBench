# phmm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME phmm
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
