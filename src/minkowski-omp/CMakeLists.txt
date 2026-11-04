# minkowski-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME minkowski
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES math
)
