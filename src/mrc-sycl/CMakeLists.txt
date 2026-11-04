# mrc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mrc
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
