# graphB+-cuda/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/graph.tar.gz")

add_hecbench_benchmark(
    NAME graphB+
    MODEL cuda
    SOURCES main.cu
    CATEGORIES graph
)
