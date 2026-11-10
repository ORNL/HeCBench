# clink-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/data.tar.gz")

add_hecbench_benchmark(
    NAME clink
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
