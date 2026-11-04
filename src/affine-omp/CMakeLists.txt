# affine-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME affine
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
