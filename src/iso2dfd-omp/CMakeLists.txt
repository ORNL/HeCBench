# iso2dfd-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME iso2dfd
    MODEL omp
    SOURCES iso2dfd.cpp
    CATEGORIES simulation
)
