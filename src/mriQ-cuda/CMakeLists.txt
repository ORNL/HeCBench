# mriQ-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/datasets.tar.bz")

add_hecbench_benchmark(
    NAME mriQ
    MODEL cuda
    SOURCES main.cu file.cc
    CATEGORIES imaging
    COMPILE_OPTIONS -use_fast_math
)
