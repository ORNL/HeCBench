# flame-cuda/CMakeLists.txt

set(DATA_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${DATA_DIR}/gold.tar.gz")

add_hecbench_benchmark(
    NAME flame
    MODEL cuda
    SOURCES main.cu utils.cu
    CATEGORIES imaging
)
