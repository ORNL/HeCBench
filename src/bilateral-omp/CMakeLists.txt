# bilateral-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bilateral
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
