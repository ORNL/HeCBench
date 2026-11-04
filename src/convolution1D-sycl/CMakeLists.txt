# convolution1D-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolution1D
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
