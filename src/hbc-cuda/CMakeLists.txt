# hbc-cuda/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/test_graph.tar.gz")

add_hecbench_benchmark(
    NAME hbc
    MODEL cuda
    SOURCES kernels.cu main.cpp parse.cpp sequential.cpp util.cpp
    CATEGORIES graph
)
