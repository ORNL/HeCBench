# interval-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME interval
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
