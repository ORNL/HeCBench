# wyllie-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME wyllie
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
