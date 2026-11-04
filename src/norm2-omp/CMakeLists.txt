# norm2-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME norm2
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
