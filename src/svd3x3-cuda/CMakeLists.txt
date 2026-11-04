# svd3x3-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/dataset.tar.bz")

add_hecbench_benchmark(
    NAME svd3x3
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    COMPILE_OPTIONS -use_fast_math
)
