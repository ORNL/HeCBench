# daphne-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME daphne
    MODEL cuda
    SOURCES
        src/common/main.cpp
        src/points2image/kernel.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    CATEGORIES automotive
)
