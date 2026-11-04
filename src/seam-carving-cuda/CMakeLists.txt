# seam-carving-cuda/CMakeLists.txt

set(DATA_DIR "${CMAKE_CURRENT_LIST_DIR}/")
UNZIPFILE("${DATA_DIR}/image.tar.gz")

add_hecbench_benchmark(
    NAME seam-carving
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
