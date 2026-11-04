# thomas-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME thomas
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
