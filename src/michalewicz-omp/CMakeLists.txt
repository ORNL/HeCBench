# michalewicz-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME michalewicz
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
