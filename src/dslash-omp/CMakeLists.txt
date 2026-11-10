# dslash-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME dslash
    MODEL omp
    SOURCES dslash.cpp kernels.cpp main.cpp
    CATEGORIES simulation
)
