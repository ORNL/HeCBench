# colorwheel-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME colorwheel
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
