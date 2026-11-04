# popcount-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME popcount
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
