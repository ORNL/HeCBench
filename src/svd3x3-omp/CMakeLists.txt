# svd3x3-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME svd3x3
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES math
)
