# matrix-rotate-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME matrix-rotate
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
