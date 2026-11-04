# nqueen-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME nqueen
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
