# divergence-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME divergence
    MODEL omp
    SOURCES divergence.cpp
    CATEGORIES math
)
