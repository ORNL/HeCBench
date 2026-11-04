# ced-cuda/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/frames.tar.gz")

add_hecbench_benchmark(
    NAME ced
    MODEL cuda
    SOURCES kernel_reference.cu main.cu
    CATEGORIES imaging
    LINK_LIBRARIES pthread
)
