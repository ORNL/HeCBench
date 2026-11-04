# gd-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gd
    MODEL cuda
    SOURCES main.cu utils.cpp
    CATEGORIES ml
)
