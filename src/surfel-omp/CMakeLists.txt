# surfel-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME surfel
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
