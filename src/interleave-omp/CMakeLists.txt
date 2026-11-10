# interleave-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME interleave
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
