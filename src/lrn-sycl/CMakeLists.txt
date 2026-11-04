# lrn-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lrn
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
