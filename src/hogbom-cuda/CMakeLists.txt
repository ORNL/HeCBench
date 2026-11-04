# hogbom-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/data.tar.bz")

add_hecbench_benchmark(
    NAME hogbom
    MODEL cuda
    SOURCES kernels.cu main.cpp reference.cpp timer.cpp
    CATEGORIES imaging
)
