# rfs-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rfs
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
