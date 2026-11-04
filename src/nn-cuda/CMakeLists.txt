# nn-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/../data/nn/nn.tar.bz")

add_hecbench_benchmark(
    NAME nn
    MODEL cuda
    SOURCES nearestNeighbor.cu utils.cu
    CATEGORIES ml
)
