# mpc-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/trace.tar.bz")

add_hecbench_benchmark(
    NAME mpc
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
