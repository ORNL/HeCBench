# matrix-rotate-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME matrix-rotate
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
