# remap-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME remap
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
