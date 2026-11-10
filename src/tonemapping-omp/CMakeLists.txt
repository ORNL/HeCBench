# tonemapping-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME tonemapping
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
