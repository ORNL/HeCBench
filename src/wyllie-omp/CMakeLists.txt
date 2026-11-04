# wyllie-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME wyllie
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
