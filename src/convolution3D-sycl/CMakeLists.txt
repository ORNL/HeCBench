# convolution3D-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolution3D
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
