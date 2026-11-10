# vmc-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME vmc
    MODEL omp
    SOURCES vmc.cpp
    CATEGORIES simulation
)
