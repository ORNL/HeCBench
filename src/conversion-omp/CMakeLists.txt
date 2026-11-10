# conversion-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME conversion
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
