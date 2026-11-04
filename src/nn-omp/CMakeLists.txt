# nn-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME nn
    MODEL omp
    SOURCES nearestNeighbor.cpp utils.cpp
    CATEGORIES ml
)
