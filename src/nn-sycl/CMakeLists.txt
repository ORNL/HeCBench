# nn-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME nn
    MODEL sycl
    SOURCES nearestNeighbor.cpp utils.cpp
    CATEGORIES ml
)
