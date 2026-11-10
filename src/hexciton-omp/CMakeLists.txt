# hexciton-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hexciton
    MODEL omp
    SOURCES main.cpp reference.cpp utils.cpp
    CATEGORIES simulation
)
