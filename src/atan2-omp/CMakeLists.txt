# atan2-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME atan2
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
