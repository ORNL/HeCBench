# matrixT-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME matrixT
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
