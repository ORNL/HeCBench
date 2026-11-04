# kmc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME kmc
    MODEL cuda
    SOURCES main.cu svm.cpp
    CATEGORIES ml
    LINK_LIBRARIES cublas
)
