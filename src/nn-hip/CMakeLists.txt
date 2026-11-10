# nn-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME nn
    MODEL hip
    SOURCES nearestNeighbor.cu utils.cu
    CATEGORIES ml
)
