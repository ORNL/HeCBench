# reverse-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME reverse
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
