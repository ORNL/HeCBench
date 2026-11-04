# lanczos-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/data.tar.bz")

add_hecbench_benchmark(
    NAME lanczos
    MODEL cuda
    SOURCES lanczos.cu main.cu
    CATEGORIES math
)
