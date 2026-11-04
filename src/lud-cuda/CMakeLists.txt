# lud-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lud
    MODEL cuda
    SOURCES lud.cu common/common.cpp 
    CATEGORIES math
    INCLUDE_DIRS common
)
