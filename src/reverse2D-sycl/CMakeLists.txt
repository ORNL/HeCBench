# reverse2D-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME reverse2D
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
