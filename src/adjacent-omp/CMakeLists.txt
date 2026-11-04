# adjacent-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME adjacent
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
