# bwt-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bwt
    MODEL cuda
    SOURCES bwt.cu main.cpp
    CATEGORIES compression
)
