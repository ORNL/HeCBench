# vol2col-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME vol2col
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
