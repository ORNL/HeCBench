# gaussian-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME gaussian
    MODEL omp
    SOURCES gaussianElim.cpp utils.cpp
    CATEGORIES math
)
